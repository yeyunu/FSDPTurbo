"""BF16 DeepGEMM adapter for all three grouped attention projection contractions."""

from functools import lru_cache
from importlib import import_module
import logging
from types import ModuleType

import torch
from torch.autograd.function import FunctionCtx, once_differentiable

from fsdp_turbo.ops.cpu.attention_projection import attention_output_projection_cpu
from fsdp_turbo.ops.registry import register_op
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_deepgemm() -> ModuleType | None:
    try:
        return import_module('deep_gemm')
    except ModuleNotFoundError as error:
        if error.name != 'deep_gemm':
            raise
        print_rank(
            logger.warning,
            'DeepGEMM Ascend is not installed; model.use_deepgemm_wo_a=True is falling back '
            'to torch.einsum on the input device.',
        )
        return None


class _DeepGemmAttentionProjection(torch.autograd.Function):
    """Differentiate the grouped projection using three BF16 contractions."""

    @staticmethod
    def forward(ctx: FunctionCtx, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project flattened tokens and save operands for the first-order backward."""
        output = inputs.new_empty((inputs.shape[0], weight.shape[0], weight.shape[1]))
        _get_deepgemm().einsum('bhr,hdr->bhd', inputs, weight, output)
        ctx.save_for_backward(inputs, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input and weight gradients without retaining cross-call buffers."""
        inputs, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_inputs, grad_weight = torch.empty_like(inputs), torch.empty_like(weight)
        backend = _get_deepgemm()
        backend.einsum('bhd,hdr->bhr', grad_output, weight, grad_inputs)
        backend.einsum('bhd,bhr->hdr', grad_output, inputs, grad_weight)
        return grad_inputs, grad_weight


@register_op('attention_output_projection', 'npu')
def attention_output_projection_npu(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run BF16 projection and first-order gradients with private output buffers.

    Materialize noncontiguous inputs and weights before saving them. Saved tensors
    follow ordinary autograd/FSDP lifetimes; no cross-call output or weight cache is
    kept. Higher-order differentiation is unsupported by the underlying kernels.
    A missing DeepGEMM package uses the differentiable reference on the same device
    and warns once on rank zero. Other import and execution failures propagate.
    """
    if inputs.device.type != 'npu' or inputs.dtype != torch.bfloat16:
        raise ValueError('DeepGEMM attention output projection requires BF16 NPU tensors')
    if _get_deepgemm() is None:
        return attention_output_projection_cpu(inputs, weight)
    batch, sequence, groups, features = inputs.shape
    projected = _DeepGemmAttentionProjection.apply(
        inputs.reshape(batch * sequence, groups, features).contiguous(), weight.contiguous()
    )
    return projected.reshape(batch, sequence, groups, weight.shape[1])
