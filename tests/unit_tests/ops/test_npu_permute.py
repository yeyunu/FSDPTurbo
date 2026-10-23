import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[3] / "fsdp_turbo" / "ops" / "npu" / "permute.py"
_SPEC = importlib.util.spec_from_file_location("test_npu_permute_impl", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
npu_permute = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(npu_permute)


class _FakeTorchNPU:
    @staticmethod
    def npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, *, restore_shape=None):
        assert permuted_tokens.dtype == probs.dtype
        return permuted_tokens


def test_unpermute_npu_preserves_activation_dtype(monkeypatch):
    monkeypatch.setattr(npu_permute, "torch_npu", _FakeTorchNPU)
    activations = torch.ones((2, 4), dtype=torch.bfloat16)
    sorted_indices = torch.arange(2, dtype=torch.int32)
    router_probs = torch.ones((2, 1), dtype=torch.float32)

    output = npu_permute.unpermute_npu(activations, sorted_indices, router_probs)

    assert output.dtype == activations.dtype
