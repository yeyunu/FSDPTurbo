from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


@register_op('swiglu', 'npu')
def fused_swiglu(inputs, dim=-1):
    return torch_npu.npu_swiglu(inputs, dim=dim)
