import weakref
from unittest.mock import Mock, patch

import pytest
import torch

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import QuantizeConfig
from fsdp_turbo.quantization.core.post_quant_weight import PostQuantWeight
from fsdp_turbo.quantization.core.pre_quant_weight import PreQuantWeight
from fsdp_turbo.quantization.core.recompute_phase import (
    MXFP8WeightPhase,
    resolve_live_post_quant_weight,
)
from fsdp_turbo.quantization.mx_formats.mx_gmm import mx_quant_group_gemm
from fsdp_turbo.quantization.mx_formats.mx_linear import mx_mm
from fsdp_turbo.quantization.mxfp8_config import (
    MXFP8LinearConfig,
    get_mxfp8linear_config,
)


def _post_quant_weight(with_backward=False):
    weight_fwd = torch.randn(64, 64, dtype=torch.bfloat16)
    scale_fwd = torch.randn(64, 2)
    weight_bwd = torch.randn(64, 64, dtype=torch.bfloat16) if with_backward else None
    scale_bwd = torch.randn(64, 2) if with_backward else None
    return PostQuantWeight(weight_fwd, scale_fwd, weight_bwd, scale_bwd, torch.bfloat16)


def test_recompute_keeps_requested_on_demand_mode():
    quant_plan = QuantizeConfig(quant_recipe="mxfp8", fsdp_low_precision_all_gather_mode="on-demand")
    engine = Mock()
    engine.model = Mock()
    engine.config.memory.recompute = True
    engine.config.quantization.quantization_plan = quant_plan
    converter = Mock()

    with patch(
        "fsdp_turbo.quantization.converter.model_converter.build_model_converter",
        return_value=converter,
    ):
        FSDPTurbo.apply_quantization_modules(engine)

    assert quant_plan.fsdp_low_precision_all_gather_mode == "on-demand"
    assert quant_plan._recompute_aware is True
    converter.convert.assert_called_once_with(engine.model)


def test_all_mode_does_not_enable_recompute_state_machine():
    quant_plan = QuantizeConfig(quant_recipe="mxfp8", fsdp_low_precision_all_gather_mode="all")
    engine = Mock()
    engine.model = Mock()
    engine.config.memory.recompute = True
    engine.config.quantization.quantization_plan = quant_plan

    with patch(
        "fsdp_turbo.quantization.converter.model_converter.build_model_converter",
        return_value=Mock(),
    ):
        FSDPTurbo.apply_quantization_modules(engine)

    assert quant_plan.fsdp_low_precision_all_gather_mode == "all"
    assert quant_plan._recompute_aware is False


def test_recompute_aware_flag_is_propagated_to_linear_config():
    quant_plan = QuantizeConfig(quant_recipe="mxfp8")
    quant_plan._recompute_aware = True

    linear_config = get_mxfp8linear_config(quant_plan)

    assert linear_config._recompute_aware is True


def test_post_quant_accessors_fail_closed_for_inactive_phase():
    weight = _post_quant_weight()

    weight.get_forward_pair()
    with pytest.raises(RuntimeError, match="backward weight is not resident"):
        weight.get_backward_pair()


def test_live_weight_is_resolved_after_recompute_before_backward():
    live_weight = _post_quant_weight()
    forward_weight = live_weight.detach()
    owner = Mock()
    owner.weight = live_weight
    backward_weight = torch.randn(64, 64, dtype=torch.bfloat16)
    backward_scale = torch.randn(64, 2)

    live_weight._weight_fwd = None
    live_weight._scale_fwd = None
    live_weight._weight_bwd = backward_weight
    live_weight._scale_bwd = backward_scale

    with pytest.raises(RuntimeError, match="backward weight is not resident"):
        forward_weight.get_backward_pair()
    with patch("fsdp_turbo.quantization.core.recompute_phase.ensure_mxfp8_backward_phase") as ensure_phase:
        resolved = resolve_live_post_quant_weight(forward_weight, weakref.ref(owner), "weight")

    ensure_phase.assert_called_once_with(owner)
    assert resolved is live_weight
    resolved_weight, resolved_scale = resolved.get_backward_pair()
    assert resolved_weight is backward_weight
    assert resolved_scale is backward_scale


def test_live_owner_shape_mismatch_fails_closed():
    saved_weight = _post_quant_weight()
    owner = Mock()
    owner.weight = PostQuantWeight(
        torch.randn(32, 64, dtype=torch.bfloat16),
        torch.randn(32, 2),
        torch.randn(32, 64, dtype=torch.bfloat16),
        torch.randn(32, 2),
        torch.bfloat16,
    )

    with (
        patch("fsdp_turbo.quantization.core.recompute_phase.ensure_mxfp8_backward_phase"),
        pytest.raises(RuntimeError, match="shape changed across recompute"),
    ):
        resolve_live_post_quant_weight(saved_weight, weakref.ref(owner), "weight")


class _FakeQuantizedTensor:
    def __init__(self, tensor):
        self.data = tensor.clone()
        self.scale = torch.ones(1, tensor.shape[-1], dtype=tensor.dtype)
        self.data_t = tensor.clone()
        self.scale_t = torch.ones(1, 1, dtype=tensor.dtype)
        self.ori_dtype = tensor.dtype


class _FakeMXTensor:
    @staticmethod
    def to_mxfp8(tensor, _key):
        return _FakeQuantizedTensor(tensor)


def test_linear_autograd_backward_uses_live_owner_weight():
    input_tensor = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
    stale_weight = _post_quant_weight()
    owner = Mock()
    owner.weight = stale_weight

    live_backward = torch.randn(64, 64, dtype=torch.bfloat16)
    live_weight = PostQuantWeight(
        torch.randn(64, 64, dtype=torch.bfloat16),
        torch.ones(64, 2, dtype=torch.bfloat16),
        live_backward,
        torch.ones(64, 2, dtype=torch.bfloat16),
        torch.bfloat16,
    )

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_linear.mx_quant_matmul",
            side_effect=lambda lhs, rhs, *_args, **_kwargs: lhs @ rhs,
        ),
        patch("fsdp_turbo.quantization.core.recompute_phase.ensure_mxfp8_backward_phase"),
    ):
        output = mx_mm.apply(input_tensor, stale_weight, _FakeMXTensor(), owner)
        owner.weight = live_weight
        output.sum().backward()

    expected_grad = torch.ones_like(output) @ live_backward
    torch.testing.assert_close(input_tensor.grad, expected_grad)


@pytest.mark.parametrize("weight_attr", ["gate_up_proj", "down_proj"])
def test_gmm_propagates_canonical_owner_attribute(weight_attr):
    input_tensor = torch.randn(4, 64, dtype=torch.bfloat16)
    post_quant_weight = _post_quant_weight()
    group_list = torch.tensor([2, 4], dtype=torch.int64)
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16
    owner = Mock()
    expected = torch.randn(4, 32, dtype=torch.bfloat16)

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(input_tensor, torch.ones(4, 2)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=expected,
        ) as grouped_matmul,
    ):
        output = mx_quant_group_gemm(
            input_tensor,
            post_quant_weight,
            group_list,
            config,
            weight_owner=owner,
            weight_attr=weight_attr,
        )

    assert output is expected
    assert grouped_matmul.call_args.args[-2] is owner
    assert grouped_matmul.call_args.args[-1] == weight_attr


def test_resolve_without_owner_fails_closed():
    """All callers must pass weight_owner; the _source fallback chain is gone."""
    saved_weight = _post_quant_weight()

    with pytest.raises(RuntimeError, match="requires weight_owner"):
        resolve_live_post_quant_weight(saved_weight, None, "weight")


def test_linear_autograd_backward_uses_plain_weight():
    """Non-post-quant branch: dx/dw come from the fp8 weight kept in ctx,
    with no owner resolution involved.
    """
    input_tensor = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, 64, dtype=torch.bfloat16, requires_grad=True)

    with patch(
        "fsdp_turbo.quantization.mx_formats.mx_linear.mx_quant_matmul",
        side_effect=lambda lhs, rhs, *_args, **_kwargs: lhs @ rhs,
    ):
        output = mx_mm.apply(input_tensor, weight, _FakeMXTensor(), None)
        output.sum().backward()

    ones = torch.ones_like(output)
    torch.testing.assert_close(input_tensor.grad, ones @ weight, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(weight.grad, ones.t() @ input_tensor.detach(), rtol=1e-2, atol=1e-2)


def test_post_quant_weight_desugar_preserves_pairs():
    """Desugar ops (clone/detach) must carry both fp8 pairs onto the new
    PostQuantWeight so backward can still find the bwd pair.
    """
    weight = _post_quant_weight(with_backward=True)

    for new_weight in (weight.clone(), weight.detach()):
        assert isinstance(new_weight, PostQuantWeight)
        assert torch.equal(new_weight._weight_fwd, weight._weight_fwd)
        assert torch.equal(new_weight._weight_bwd, weight._weight_bwd)
        assert new_weight._scale_fwd is weight._scale_fwd
        assert new_weight._scale_bwd is weight._scale_bwd


def _pre_quant_weight(config=None):
    tensor = torch.randn(64, 64, dtype=torch.bfloat16, requires_grad=True)
    return PreQuantWeight(tensor, lambda t: t, config, tensor.dtype, "test.weight")


def _fsdp_state_mock(reshard_after_forward=True):
    fsdp_state = Mock()
    fsdp_state._fsdp_param_group._reshard_after_forward = reshard_after_forward
    return fsdp_state


@pytest.mark.parametrize(
    "mode,phase,expected,exp_fwd,exp_bwd",
    [
        ("on-demand", MXFP8WeightPhase.BACKWARD, ("wb",), False, True),
        ("on-demand", MXFP8WeightPhase.FORWARD, ("wf",), True, False),
        ("all", MXFP8WeightPhase.BACKWARD, ("wf", "wb"), True, True),
    ],
)
def test_pre_quant_all_gather_selects_pairs(mode, phase, expected, exp_fwd, exp_bwd):
    """LP-AG gathers only the pair the current phase needs (on-demand) or
    both (all mode).
    """
    pq = _pre_quant_weight(config=MXFP8LinearConfig(fsdp_low_precision_all_gather_mode=mode))
    wf, sf, wb, sb = (
        torch.randn(64, 64),
        torch.randn(1, 64),
        torch.randn(64, 64),
        torch.randn(1, 64),
    )

    with (
        patch(
            "fsdp_turbo.quantization.core.pre_quant_weight.get_module_fsdp_state",
            return_value=_fsdp_state_mock(),
        ),
        patch(
            "fsdp_turbo.quantization.core.pre_quant_weight.select_recompute_aware_phase",
            return_value=phase,
        ),
        patch(
            "fsdp_turbo.quantization.core.pre_quant_weight.cached_weight_quant",
            return_value=(wf, sf, wb, sb),
        ),
    ):
        tensors, metadata = pq.fsdp_pre_all_gather(None, None, None, Mock(), None)

    fwd_usage, bwd_usage, scale_fwd, scale_bwd = metadata
    assert tensors == tuple({"wf": wf, "wb": wb}[t] for t in expected)
    assert fwd_usage is exp_fwd and bwd_usage is exp_bwd
    assert scale_fwd is sf and scale_bwd is sb


def test_pre_quant_post_all_gather_assembles_both_pairs():
    pq = _pre_quant_weight()
    wf = torch.randn(64, 64)
    wb = torch.randn(64, 64)
    metadata = (True, True, torch.randn(1, 64), torch.randn(1, 64))

    out, _ = pq.fsdp_post_all_gather((wf, wb), metadata, torch.bfloat16)

    assert isinstance(out, PostQuantWeight)
    assert out._weight_fwd is wf and out._weight_bwd is wb
    assert out._scale_fwd is metadata[2] and out._scale_bwd is metadata[3]


def test_pre_quant_post_all_gather_inplace_clears_inactive_pair():
    """out= existing PostQuantWeight: the inactive pair is nulled so a stale
    alias cannot consume a buffer from the wrong phase.
    """
    pq = _pre_quant_weight()
    wf = torch.randn(64, 64)
    metadata = (True, False, torch.randn(1, 64), None)
    out = PostQuantWeight(wf, metadata[2], torch.randn(64, 64), torch.randn(1, 64), torch.bfloat16)

    ret, _ = pq.fsdp_post_all_gather((wf,), metadata, torch.bfloat16, out=out)

    assert ret is out
    assert out._weight_fwd is wf and out._weight_bwd is None
    assert out._scale_bwd is None
