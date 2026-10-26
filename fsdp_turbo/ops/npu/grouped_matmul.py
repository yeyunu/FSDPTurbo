import torch
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weights, weights_bias, m_split, group_list_type) -> torch.Tensor:
        # Due to ascend gmm kernel k split limitations, we need a tensor m_split, not a tensor List.
        if not isinstance(m_split, torch.Tensor):
            ctx.group_list = torch.tensor(m_split, device='npu', dtype=torch.int64)
        else:
            ctx.group_list = m_split

        ctx.group_list_type = group_list_type

        # Get weight chunks
        weight_chunks = [w[0] for w in weights.chunk(weights.shape[0], dim=0)]

        # Weights follow the linear-layer convention [output_dim, input_dim] per expert.
        # Transpose to [input_dim, output_dim] so that input [M, K] @ weight.T [K, N] = output [M, N].
        # Always transpose rather than guessing from shape, since shape-based detection is
        # unreliable when input_dim == output_dim (square weight matrices).
        # With one x/weight/y, split-M selects the packed-weight API:
        # its sole weight must keep the expert axis [1, K, N].
        weights_for_matmul = [weights.transpose(1, 2)] if weights.shape[0] == 1 else [w.T for w in weight_chunks]

        ctx.save_for_backward(input_tensor, weights)

        fwd_output = torch_npu.npu_grouped_matmul(
            [input_tensor],
            weights_for_matmul,
            bias=weights_bias,
            group_list=ctx.group_list,
            split_item=2,
            group_type=0,
            group_list_type=ctx.group_list_type,
        )[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        group_list = ctx.group_list
        inp, weights = ctx.saved_tensors
        group_list_type = ctx.group_list_type

        # Get weight chunks (original [output_dim, input_dim] format)
        weight_chunks = [weights] if weights.shape[0] == 1 else [w[0] for w in weights.chunk(weights.shape[0], dim=0)]

        # Calculate input gradient: grad_output [M, N] @ weight [N, K] = grad_input [M, K].
        # Forward used transposed weights, so the original weights are used directly here.
        grad = torch_npu.npu_grouped_matmul(
            [grad_output],
            weight_chunks,
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
        )[0]

        # Calculate weight gradient (K split gmm): grad_weight = inp^T @ grad_output = [K, N]
        grad_weight = torch_npu.npu_grouped_matmul(
            [inp.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=group_list_type,
        )[0]

        # Preserve the GEMM and reduction order, but return the [E,N,K] view.
        # AccumulateGrad/FSDP can copy it into its destination layout directly;
        # eagerly stacking per-expert transposes adds a separate materialization.
        return grad, grad_weight.transpose(-1, -2), None, None, None


@register_op('grouped_matmul', 'npu')
def grouped_matmul_npu(inputs, m_split, weights):
    return GroupedMatmul.apply(inputs, weights, None, m_split, 1)


@register_op('w4a16_grouped_matmul', 'npu')
def w4a16_grouped_matmul_npu(
    x,
    weights,
    bias=None,
    group_list=None,
    group_type=0,
    output_dtype=None,
    group_list_type=0,
    split_item=2,
):
    """Low-precision (W4A16) grouped matmul on Ascend.

    Wraps ``torch_npu.npu_grouped_matmul`` with the W4A16 QAT recipe's
    argument set (``output_dtype``, ``group_type``, ``split_item``).
    """
    return torch_npu.npu_grouped_matmul(
        [x],
        weights,
        bias=bias,
        group_list=group_list,
        group_type=group_type,
        output_dtype=output_dtype,
        group_list_type=group_list_type,
        split_item=split_item,
    )
