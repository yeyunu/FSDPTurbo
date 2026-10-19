"""Reference clipped SwiGLU, also usable on accelerator tensors."""

import torch.nn.functional as F

from fsdp_turbo.ops.registry import register_op


@register_op("clipped_swiglu", "cpu")
def clipped_swiglu_cpu(x, limit):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
