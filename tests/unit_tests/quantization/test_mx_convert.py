import pytest
import torch

from fsdp_turbo.quantization.core.pre_quant_weight import PreQuantWeight
from fsdp_turbo.quantization.mx_formats.mx_gmm import MXFP8GMM, weight_quant_gmm
from fsdp_turbo.quantization.mx_formats.mx_linear import MXLinear, weight_quant
from fsdp_turbo.quantization.mxfp8_config import MXFP8LinearConfig


class _FakeGMMModule:
    """Stand-in for an HF MoE experts module with gate_up_proj/down_proj."""

    def __init__(self, gate_up, down, num_experts, hidden_dim=None, intermediate_size=None):
        self.gate_up_proj = gate_up
        self.down_proj = down
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self._quant_config = None


def _gmm_config(lp_ag=True, **kwargs):
    return MXFP8LinearConfig(enable_fsdp_low_precision_all_gather=lp_ag, **kwargs)


def test_gmm_from_float_lp_ag_wraps_weights_in_pre_quant():
    """LP-AG path: a fresh module whose weights are PreQuantWeight wrapping
    the original tensors with the weight_quant_gmm quantizer.
    """
    e, k, n = 2, 4, 8
    gate_up = torch.randn(e * k, 2 * n, dtype=torch.bfloat16)
    down = torch.randn(e * n, k, dtype=torch.bfloat16)
    mod = _FakeGMMModule(gate_up, down, e, hidden_dim=k, intermediate_size=n)

    new_mod = MXFP8GMM.from_float(mod, config=_gmm_config())

    assert new_mod is not mod
    assert isinstance(new_mod, MXFP8GMM)
    assert new_mod.num_experts == e
    assert new_mod.hidden_dim == k
    assert new_mod.intermediate_size == n

    for attr, shape in (("gate_up_proj", gate_up.shape), ("down_proj", down.shape)):
        pq = getattr(new_mod, attr)
        assert isinstance(pq, PreQuantWeight)
        assert pq._tensor is getattr(mod, attr)  # original tensor, not a copy
        assert pq.shape == shape
        assert pq._quantizer.func is weight_quant_gmm
        assert pq._quantizer.keywords["dst_type"] == new_mod.config.get_key_dtype("weight")


def test_gmm_from_float_derives_dims_from_2d_layouts():
    """Without hidden/intermediate attrs, dims come from the gate_up shape:
    [E*K, 2N] (dim0 < dim1) and [E*2N, K] (dim0 > dim1) both resolve.
    """
    e, k, n = 2, 4, 8

    for gate_up_shape in ((e * k, 2 * n), (e * 2 * n, k)):
        mod = _FakeGMMModule(
            torch.randn(*gate_up_shape, dtype=torch.bfloat16),
            torch.randn(e * n, k, dtype=torch.bfloat16),
            e,
        )
        new_mod = MXFP8GMM.from_float(mod, config=_gmm_config())

        assert new_mod.hidden_dim == k
        assert new_mod.intermediate_size == n
        new_shape = new_mod.gate_up_proj._quantizer.keywords["new_shape"]
        assert new_shape == (-1, k, 2 * n)


def test_gmm_from_float_preserves_3d_weights():
    """3D [E, N, K] weights (HF layout) are wrapped as-is, not reshaped."""
    e, k, n = 2, 4, 8
    gate_up = torch.randn(e, 2 * n, k, dtype=torch.bfloat16)
    down = torch.randn(e, k, n, dtype=torch.bfloat16)
    mod = _FakeGMMModule(gate_up, down, e, hidden_dim=k, intermediate_size=n)

    new_mod = MXFP8GMM.from_float(mod, config=_gmm_config())

    assert new_mod.gate_up_proj.shape == gate_up.shape
    assert new_mod.down_proj.shape == down.shape
    assert new_mod.gate_up_proj._tensor is gate_up


def test_gmm_from_float_patches_class_without_lp_ag():
    """Without LP-AG the module is converted in place via class patch."""
    e, k, n = 2, 4, 8
    mod = _FakeGMMModule(
        torch.randn(e * k, 2 * n, dtype=torch.bfloat16),
        torch.randn(e * n, k, dtype=torch.bfloat16),
        e,
        hidden_dim=k,
        intermediate_size=n,
    )

    MXFP8GMM.from_float(mod, config=_gmm_config(lp_ag=False))

    assert isinstance(mod, MXFP8GMM)
    assert mod._quant_config is not None
    assert not isinstance(mod.gate_up_proj, PreQuantWeight)


@pytest.mark.parametrize(
    "lp_ag,fsdp_world_size,out_features,expect_wrapped",
    [
        (True, 1, 32, True),  # LP-AG: wrapped with the dual-axis quantizer
        (True, 2, 5, False),  # out % world_size != 0 -> class-patch fallback
        (False, 1, 32, False),  # no LP-AG: class patch only
    ],
)
def test_linear_from_float(lp_ag, fsdp_world_size, out_features, expect_wrapped):
    linear = torch.nn.Linear(16, out_features, bias=True, dtype=torch.bfloat16)
    linear.weight.requires_grad = True

    new_mod = MXLinear.from_float(linear, config=_gmm_config(lp_ag=lp_ag, fsdp_world_size=fsdp_world_size))

    assert isinstance(new_mod, MXLinear)
    assert isinstance(new_mod.weight, PreQuantWeight) is expect_wrapped
    if expect_wrapped:
        assert new_mod.weight._tensor is linear.weight
        assert new_mod.weight._quantizer.func is weight_quant
        assert new_mod.bias is linear.bias
        assert new_mod.config is not None


def test_config_get_key_dtype_dispatches_by_key():
    """E4M3 default: all keys e4m3; HYBRID: grads become e5m2."""
    e4m3 = MXFP8LinearConfig().get_key_dtype
    assert e4m3("inputs") == e4m3("weight") == e4m3("grads") == torch.float8_e4m3fn

    hybrid = MXFP8LinearConfig(quant_format="HYBRID").get_key_dtype
    assert hybrid("inputs") == hybrid("weight") == torch.float8_e4m3fn
    assert hybrid("grads") == torch.float8_e5m2
