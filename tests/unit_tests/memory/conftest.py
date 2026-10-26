"""Session-wide torch.accelerator injection for device-backed tests.

SwapCache/act-stash/op-replay call torch.accelerator Stream/Event/RNG entry
points, which only exist after set_accelerator_compatible() proxies them from
torch.npu/torch.cuda. On CPU-only
hosts the injection raises and device tests skip themselves via skipif.
"""

try:
    from fsdp_turbo.utils.device import set_accelerator_compatible

    set_accelerator_compatible()
except Exception:
    pass
