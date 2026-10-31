"""Grouped attention output projection with an optional DeepGEMM backend."""

from importlib import import_module

import torch

from fsdp_turbo.ops.registry import dispatch_op


def attention_output_projection(
    inputs: torch.Tensor, weight: torch.Tensor, *, use_deepgemm: bool = False
) -> torch.Tensor:
    """Project ``[B,S,G,D]`` inputs with ``[G,R,D]`` weights to ``[B,S,G,R]``.

    Default execution preserves torch einsum and its autograd. Explicitly enabled
    NPU execution uses BF16 DeepGEMM when installed; a missing package warns once
    on rank zero and uses the differentiable reference. Other devices and empty
    inputs also use the reference. No quantization or precision conversion is applied.
    The reference runs on the tensors' existing device; it does not copy to CPU.

    Raises:
        ValueError: Input shapes, devices or dtypes violate the projection contract.
    """
    if inputs.ndim != 4 or weight.ndim != 3 or inputs.shape[-2:] != (weight.shape[0], weight.shape[2]):
        raise ValueError("attention output projection requires [B,S,G,D] inputs and [G,R,D] weights")
    if inputs.device != weight.device or inputs.dtype != weight.dtype or not inputs.is_floating_point():
        raise ValueError("attention output projection requires floating tensors with the same dtype and device")
    backend = 'npu' if use_deepgemm and inputs.device.type == 'npu' and inputs.numel() and weight.numel() else 'cpu'
    if backend == 'npu':
        # Keep device-backend import failures visible; the adapter handles only
        # an absent DeepGEMM package before entering its custom autograd function.
        import_module('fsdp_turbo.ops.npu.attention_projection')
    return dispatch_op('attention_output_projection', inputs, weight, device_type=backend)
