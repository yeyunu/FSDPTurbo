"""SwiGLU activation with an NPU fast path and unchanged eager arithmetic."""

import torch
import torch.nn.functional as F

from fsdp_turbo.ops.quantize import swiglu


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fuse already-clamped branches without changing their dtype."""
    if gate.device.type == "npu" and gate.numel() > 0:
        return swiglu(torch.cat((gate, up), dim=-1), dim=-1)
    return F.silu(gate) * up


def packed_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """Use a packed projection directly; empty tensors retain autograd support."""
    if gate_up.device.type == "npu" and gate_up.numel() > 0:
        return swiglu(gate_up.contiguous(), dim=-1)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up
