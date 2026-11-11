"""Session-wide torch.accelerator injection for device-backed tests.

SwapCache/act-stash/op-replay call torch.accelerator Stream/Event/RNG entry
points, which only exist after set_accelerator_compatible() proxies them from
torch.npu/torch.cuda. On CPU-only
hosts the injection raises and device tests skip themselves via skipif.
"""

import pytest

try:
    from fsdp_turbo.utils.device import set_accelerator_compatible

    set_accelerator_compatible()
except Exception:
    pass


@pytest.fixture(autouse=True)
def _isolate_replay_context_registry():
    """Hide the replay context model adapters register at import time.

    ``test_recompute.py`` asserts the exact ``checkpoint`` kwargs, so it must
    see an empty registry no matter what else the session collected.
    """
    try:
        from fsdp_turbo.memory.recompute import recompute
    except ImportError:
        yield
        return
    saved = list(recompute._replay_context_fns)
    recompute._replay_context_fns[:] = []
    try:
        yield
    finally:
        recompute._replay_context_fns[:] = saved
