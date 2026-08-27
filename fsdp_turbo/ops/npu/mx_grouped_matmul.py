# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import weakref

import torch
from torch.distributed.tensor import DTensor

try:
    import torch_npu
except ImportError:
    torch_npu = None

from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.quantization.core.recompute_phase import resolve_live_post_quant_weight
from fsdp_turbo.utils.gmm_layout import to_gmm_3d


@torch._dynamo.allow_in_graph
class MxGroupedMatmul(torch.autograd.Function):
    """FP8 grouped matmul on NPU with pre-quantized inputs."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        weight_fwd: torch.Tensor,
        weight_scale_fwd: torch.Tensor,
        weight_bwd: torch.Tensor,
        weight_scale_bwd: torch.Tensor,
        weight_2d_shape: tuple,
        grad_fp8_dtype,
        input_fp8_dtype,
        input_dtype,
        group_list: torch.Tensor,
        group_list_type: int,
        weight_owner=None,
        weight_attr=None,
    ):
        ctx.weight_2d_shape = weight_2d_shape
        ctx.grad_fp8_dtype = grad_fp8_dtype
        ctx.input_fp8_dtype = input_fp8_dtype
        ctx.input_dtype = input_dtype
        ctx.group_list_type = group_list_type

        w = weight
        if isinstance(w, torch.nn.Parameter) and not hasattr(w, "_weight_fwd"):
            w = w.data
        if hasattr(w, "to_local"):
            w = w.to_local()
        ctx.post_quant = hasattr(w, "_weight_fwd")
        ctx.weight_ref = weight
        ctx.weight_owner_ref = weakref.ref(weight_owner) if weight_owner is not None else None
        ctx.weight_attr = weight_attr

        ctx.save_for_backward(x, group_list, weight_bwd, weight_scale_bwd)

        output = torch_npu.npu_grouped_matmul(
            [x_fp8],
            [weight_fwd],
            scale=[weight_scale_fwd],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=input_dtype,
            group_list_type=group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, group_list, weight_bwd, weight_scale_bwd = ctx.saved_tensors

        if ctx.post_quant:
            w = resolve_live_post_quant_weight(
                ctx.weight_ref,
                ctx.weight_owner_ref,
                ctx.weight_attr,
            )
            weight_bwd, weight_scale_bwd = w.get_backward_pair()
            # dx: grad [T, N] x W [E, N, K] - [E, K, N] transposed
            weight_bwd = to_gmm_3d(weight_bwd, len(group_list)).transpose(1, 2).contiguous()
            weight_scale_bwd = weight_scale_bwd.transpose(1, 2).contiguous()

        group_list_int32 = group_list.to(torch.int32)
        total_tokens = group_list[-1].item()

        grad_fp8, grad_scale = torch_npu.npu_dynamic_mx_quant(grad_output, axis=-1, dst_type=ctx.grad_fp8_dtype)

        dx = torch_npu.npu_grouped_matmul(
            [grad_fp8],
            [weight_bwd],
            scale=[weight_scale_bwd],
            per_token_scale=[grad_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=grad_output.dtype,
            group_list_type=ctx.group_list_type,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        if total_tokens == 0:
            dw = torch.zeros(ctx.weight_2d_shape, dtype=ctx.input_dtype, device=x.device)
        else:
            x_q, x_q_scale = torch_npu.npu_grouped_dynamic_mx_quant(
                x,
                group_list_int32,
                round_mode="rint",
                dst_type=ctx.input_fp8_dtype,
                blocksize=32,
            )
            grad_q, grad_q_scale = torch_npu.npu_grouped_dynamic_mx_quant(
                grad_output,
                group_list_int32,
                round_mode="rint",
                dst_type=ctx.grad_fp8_dtype,
                blocksize=32,
            )
            dw = torch_npu.npu_grouped_matmul(
                [x_q.t()],
                [grad_q],
                scale=[grad_q_scale],
                per_token_scale=[x_q_scale.transpose(0, 1)],
                group_list=group_list,
                group_type=2,
                output_dtype=ctx.input_dtype,
                group_list_type=ctx.group_list_type,
                scale_dtype=torch_npu.float8_e8m0fnu,
                per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                split_item=3,
            )[0]

        # 3D stored [E, N, K] needs transpose back; 2D [E*K, N] reshapes directly
        if len(ctx.weight_2d_shape) == 3:
            dw = dw.transpose(1, 2).reshape(ctx.weight_2d_shape)
        else:
            dw = dw.reshape(ctx.weight_2d_shape)

        if ctx.post_quant and hasattr(ctx.weight_ref, "to_local"):
            dw = DTensor.from_local(dw, ctx.weight_ref.device_mesh, ctx.weight_ref.placements)

        return (
            dx,
            dw,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@register_op('mx_grouped_matmul', 'npu')
def mx_grouped_matmul_npu(
    x,
    weight,
    x_fp8,
    x_scale,
    weight_fwd,
    weight_scale_fwd,
    weight_bwd,
    weight_scale_bwd,
    weight_2d_shape,
    grad_fp8_dtype,
    input_fp8_dtype,
    input_dtype,
    group_list,
    group_list_type,
    weight_owner=None,
    weight_attr=None,
):
    return MxGroupedMatmul.apply(
        x,
        weight,
        x_fp8,
        x_scale,
        weight_fwd,
        weight_scale_fwd,
        weight_bwd,
        weight_scale_bwd,
        weight_2d_shape,
        grad_fp8_dtype,
        input_fp8_dtype,
        input_dtype,
        group_list,
        group_list_type,
        weight_owner,
        weight_attr,
    )
