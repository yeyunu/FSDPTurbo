from contextlib import ExitStack, contextmanager
from unittest.mock import Mock, patch

import pytest
import torch

# Register the pure-torch op fallbacks so real quantization runs without NPU.
import fsdp_turbo.ops.cpu.mx_quant  # noqa: F401
from fsdp_turbo.quantization.core.post_quant_weight import PostQuantWeight
from fsdp_turbo.quantization.mx_formats.mx_gmm import (
    MXFP8GMM,
    mx_quant_group_gemm,
    weight_quant_gmm,
)
from fsdp_turbo.utils.gmm_layout import to_gmm_3d


def _device():
    """Run tensors on the active accelerator so real quantization dispatches
    to the NPU implementation there; plain CPU hosts use CPU tensors.
    """
    accelerator = torch.accelerator.current_accelerator()
    return accelerator.type if accelerator is not None else "cpu"


def _quantizer_mock(captured):
    def fake_dual_axis(tensor, dst_type=None):
        captured["input"] = tensor
        return (
            tensor.clone(),
            torch.ones_like(tensor),
            tensor.clone(),
            torch.ones_like(tensor),
        )

    return fake_dual_axis


def test_weight_quant_gmm_2d_quantizes_in_op_layout_no_transpose():
    """2D [E*K, N]: the dual-axis quantizer must see [E, K, N] as a plain view
    (no transpose) and outputs must be stored back in the original 2D shape.
    """
    e, k, n = 4, 8, 6
    weight = torch.randn(e * k, n)
    captured = {}

    with patch(
        "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
        side_effect=_quantizer_mock(captured),
    ):
        weight_fwd, _, weight_bwd, _ = weight_quant_gmm(weight, torch.bfloat16, (-1, k, n), quant_format="E4M3")

    assert torch.equal(captured["input"], weight.view(e, k, -1))  # plain view, no transpose
    assert captured["input"].shape == (e, k, n)
    assert weight_fwd.shape == weight.shape and torch.equal(weight_fwd, weight)
    assert weight_bwd.shape == weight.shape


def test_weight_quant_gmm_3d_transposes_to_op_layout():
    """3D [E, N, K]: the quantizer must transpose to [E, K, N] and store back."""
    e, k, n = 4, 8, 6
    weight = torch.randn(e, n, k)
    captured = {}

    with patch(
        "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
        side_effect=_quantizer_mock(captured),
    ):
        weight_fwd, _, weight_bwd, _ = weight_quant_gmm(weight, torch.bfloat16, (-1, k, n), quant_format="E4M3")

    assert torch.equal(captured["input"], weight.transpose(1, 2))
    assert weight_fwd.shape == weight.shape and torch.equal(weight_fwd, weight)
    assert weight_bwd.shape == weight.shape


def test_storage_layout_round_trip_preserves_quantized_op_layout():
    """Quantized [E, K, N] stored in the original layout must re-view losslessly:
    to_gmm_3d recovers exactly the tensor the dual-axis quantizer saw.
    """
    e, k, n = 4, 8, 6

    for stored_shape in ((e * k, n), (e, n, k), (e, k, n)):
        captured = {}
        weight = torch.randn(*stored_shape)
        with patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
            side_effect=_quantizer_mock(captured),
        ):
            weight_fwd, _, weight_bwd, _ = weight_quant_gmm(weight, torch.bfloat16, (-1, k, n), quant_format="E4M3")

        assert torch.equal(to_gmm_3d(weight_fwd, e, k=k), captured["input"])
        assert torch.equal(to_gmm_3d(weight_bwd, e, k=k), captured["input"])


def test_weight_quant_gmm_3d_op_layout_quantizes_without_transpose():
    """3D [E, K, N] (op layout, MindSpeed-MM style): no transpose in or out."""
    e, k, n = 4, 8, 6
    weight = torch.randn(e, k, n)
    captured = {}

    with patch(
        "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
        side_effect=_quantizer_mock(captured),
    ):
        weight_fwd, _, weight_bwd, _ = weight_quant_gmm(weight, torch.bfloat16, (-1, k, n), quant_format="E4M3")

    assert captured["input"] is not None
    assert captured["input"].shape == (e, k, n)
    assert torch.equal(captured["input"], weight)  # no transpose
    assert weight_fwd.shape == weight.shape and torch.equal(weight_fwd, weight)
    assert weight_bwd.shape == weight.shape and torch.equal(weight_bwd, weight)


def _post_quant_weight(fwd_shape, bwd_shape=None):
    weight_fwd = torch.randn(fwd_shape, dtype=torch.bfloat16)
    scale_fwd = torch.ones(fwd_shape[-1], dtype=torch.bfloat16)
    weight_bwd = torch.randn(bwd_shape, dtype=torch.bfloat16) if bwd_shape else None
    scale_bwd = torch.ones(bwd_shape[-1], dtype=torch.bfloat16) if bwd_shape else None
    return PostQuantWeight(weight_fwd, scale_fwd, weight_bwd, scale_bwd, torch.bfloat16)


def test_post_quant_path_views_fwd_pair_in_gmm_layout():
    """is_post_quant: the stored fwd pair must be converted to [E, K, N] and
    the bwd pair passed as None (resolved live in backward).
    """
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    pq = _post_quant_weight(fwd_shape=(e * k, n))
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(x, torch.ones(8, 1)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, pq, group_list, config, weight_owner=Mock(), weight_attr="down_proj")

    weight_fwd = grouped_matmul.call_args.args[4]
    weight_bwd = grouped_matmul.call_args.args[6]
    assert weight_fwd.shape == (e, k, n)
    assert torch.equal(weight_fwd, pq._weight_fwd.view(e, k, -1))
    assert weight_bwd is None


def test_post_quant_path_keeps_3d_op_layout_storage():
    """is_post_quant with 3D [E, K, N] stored: fwd pair reaches kernel as-is."""
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    pq = _post_quant_weight(fwd_shape=(e, k, n))
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(x, torch.ones(8, 1)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, pq, group_list, config, weight_owner=Mock(), weight_attr="down_proj")

    weight_fwd = grouped_matmul.call_args.args[4]
    assert weight_fwd.shape == (e, k, n)
    assert torch.equal(weight_fwd, pq._weight_fwd)  # stored as-is, no transpose


def test_plain_weight_3d_op_layout_quantizes_without_transpose():
    """else path with 3D [E, K, N] stored: fwd stays [E, K, N], bwd transposed."""
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    weight = torch.randn(e, k, n, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16
    captured = {}

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(x, torch.ones(8, 1)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
            side_effect=_quantizer_mock(captured),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, weight, group_list, config)

    assert captured["input"].shape == (e, k, n)  # no transpose for op layout
    weight_fwd = grouped_matmul.call_args.args[4]
    weight_bwd = grouped_matmul.call_args.args[6]
    assert weight_fwd.shape == (e, k, n)
    assert weight_bwd.shape == (e, n, k)
    assert torch.equal(weight_bwd, weight.transpose(1, 2))


def test_plain_weight_grad_enabled_quantizes_bwd_pair_transposed():
    """else path with grad_enabled: fwd pair [E, K, N] and bwd pair transposed
    to [E, N, K] for the dx matmul.
    """
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    weight = torch.randn(e * k, n, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16
    captured = {}

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            return_value=(x, torch.ones(8, 1)),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant_with_dual_axis",
            side_effect=_quantizer_mock(captured),
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, weight, group_list, config)

    assert captured["input"].shape == (e, k, n)  # no transpose for 2D
    weight_fwd = grouped_matmul.call_args.args[4]
    weight_bwd = grouped_matmul.call_args.args[6]
    assert weight_fwd.shape == (e, k, n)
    assert weight_bwd.shape == (e, n, k)
    assert torch.equal(weight_bwd, weight.view(e, k, -1).transpose(1, 2))


def test_plain_weight_inference_skips_backward_pair():
    """else path without grad_enabled: only fwd quantization, empty bwd pair."""
    e, k, n = 4, 8, 6
    x = torch.randn(8, k, dtype=torch.bfloat16)
    weight = torch.randn(e * k, n, dtype=torch.bfloat16)
    group_list = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
    config = Mock()
    config.get_key_dtype.return_value = torch.bfloat16

    def fake_quant(tensor, axis=-1, dst_type=None):
        if axis == -2:  # weight quantization in the else branch
            return torch.zeros(e, k, n, dtype=torch.bfloat16), torch.ones(e, k, 1)
        return x, torch.ones(8, 1)  # x quantization

    with (
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.dynamic_mx_quant",
            side_effect=fake_quant,
        ),
        patch(
            "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
            return_value=torch.zeros(8, n, dtype=torch.bfloat16),
        ) as grouped_matmul,
    ):
        mx_quant_group_gemm(x, weight, group_list, config, grad_enabled=False)

    weight_bwd = grouped_matmul.call_args.args[6]
    weight_scale_bwd = grouped_matmul.call_args.args[7]
    assert weight_bwd.numel() == 0
    assert weight_scale_bwd.numel() == 0


@pytest.mark.parametrize("entry", ["ep_forward", "forward"])
def test_mxfp8_gmm_propagates_owner_and_attr(entry):
    """Real MXFP8GMM wiring: both GMMs must carry the module as weight_owner
    and the canonical attribute names (real quantization runs, only the
    kernel/swiglu/permute are mocked).
    """
    hidden, intermediate, num_experts = 64, 128, 2
    config = Mock()
    config.get_key_dtype.return_value = torch.float8_e4m3fn
    module = MXFP8GMM(
        config=config,
        num_experts=num_experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
    )
    module.gate_up_proj = torch.nn.Parameter(
        torch.randn(
            num_experts * hidden,
            intermediate * 2,
            device=_device(),
            dtype=torch.bfloat16,
        )
    )
    module.down_proj = torch.nn.Parameter(
        torch.randn(num_experts * intermediate, hidden, device=_device(), dtype=torch.bfloat16)
    )

    x = torch.randn(4, hidden, device=_device(), dtype=torch.bfloat16)
    fc2_input = torch.randn(4, intermediate, device=_device(), dtype=torch.bfloat16)
    permuted = torch.randn(4, hidden, device=_device(), dtype=torch.bfloat16)
    selected_experts = torch.tensor([0, 0, 1, 1], device=_device())

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "fsdp_turbo.distributed.expert_parallel.dispatcher_mx.swiglu",
                return_value=fc2_input,
            )
        )
        grouped_matmul = stack.enter_context(
            patch(
                "fsdp_turbo.quantization.mx_formats.mx_gmm.mx_grouped_matmul",
                return_value=torch.zeros(4, 64, device=_device(), dtype=torch.bfloat16),
            )
        )
        if entry == "forward":
            stack.enter_context(
                patch(
                    "fsdp_turbo.ops.moe.permute",
                    return_value=(permuted, torch.arange(4)),
                )
            )
            stack.enter_context(
                patch(
                    "fsdp_turbo.ops.moe.unpermute",
                    return_value=torch.zeros(4, hidden, device=_device(), dtype=torch.bfloat16),
                )
            )
            module.forward(
                x,
                selected_experts=selected_experts,
                routing_weights=torch.ones(4, device=_device(), dtype=torch.float32),
            )
        else:
            module.ep_forward(x, [2, 2])

    assert grouped_matmul.call_count == 2
    fc1 = grouped_matmul.call_args_list[0]
    fc2 = grouped_matmul.call_args_list[1]
    assert fc1.args[-2] is module
    assert fc1.args[-1] == "gate_up_proj"
    assert fc2.args[-2] is module
    assert fc2.args[-1] == "down_proj"


# ============================================================================
# ep_forward_fused / ep_forward_eager tests
# ============================================================================

_DISPATCHER = 'fsdp_turbo.distributed.expert_parallel.dispatcher'
_DISPATCHER_MX = 'fsdp_turbo.distributed.expert_parallel.dispatcher_mx'
_MX_GMM = 'fsdp_turbo.quantization.mx_formats.mx_gmm'

# Shape constants shared by all EP tests.
_EP_EXPERTS = 2
_EP_HIDDEN = 8
_EP_INTER = 16
_EP_TOKENS = 4


def _make_ep_module(is_situ=False):
    """Return an MXFP8GMM configured for EP dispatch unit tests.

    No process group is needed; distributed calls are stubbed out by _ep_stubs.
    act_fn for is_situ=True halves the last dimension, mimicking gate+up -> intermediate.
    """
    config = Mock()
    config.get_key_dtype.return_value = torch.float8_e4m3fn
    config.quant_format = 'E4M3'
    act_fn = Mock(side_effect=lambda x: x[..., : x.shape[-1] // 2]) if is_situ else None
    module = MXFP8GMM(
        config=config,
        num_experts=_EP_EXPERTS,
        hidden_size=_EP_HIDDEN,
        moe_intermediate_size=_EP_INTER,
        act_fn=act_fn,
    )
    module.gate_up_proj = torch.nn.Parameter(torch.randn(_EP_EXPERTS * _EP_HIDDEN, _EP_INTER * 2, dtype=torch.bfloat16))
    module.down_proj = torch.nn.Parameter(torch.randn(_EP_EXPERTS * _EP_INTER, _EP_HIDDEN, dtype=torch.bfloat16))
    module._is_situ = is_situ
    module._num_local_experts = _EP_EXPERTS
    module.num_global_experts = _EP_EXPERTS
    module.expert_ids_per_ep_rank = torch.zeros(_EP_EXPERTS, dtype=torch.int32)
    return module


@contextmanager
def _ep_stubs(tokens_per_expert=None):
    """Stub all distributed EP primitives and the FP8 GMM kernel.

    Yields (alltoall_combine_mock, mx_quant_group_gemm_mock).

    The GMM stub returns [total, 2*INTER] on the first call (gate-up) and
    [total, HIDDEN] on the second call (down), matching the real shapes.
    tokens_per_expert defaults to an even split across _EP_EXPERTS.
    """
    if tokens_per_expert is None:
        tokens_per_expert = [_EP_TOKENS // _EP_EXPERTS] * _EP_EXPERTS
    expert_token_counts = torch.tensor(tokens_per_expert, dtype=torch.int64)
    total = int(expert_token_counts.sum())

    indices = (expert_token_counts, torch.zeros(total, dtype=torch.int32))
    split_sizes = (
        torch.ones(_EP_EXPERTS, dtype=torch.int64),
        torch.ones(_EP_EXPERTS, dtype=torch.int64),
    )
    dispatched = torch.zeros(total, _EP_HIDDEN, dtype=torch.bfloat16)
    unpermute = (torch.arange(total), torch.arange(total))
    combined = torch.zeros(_EP_TOKENS, _EP_HIDDEN, dtype=torch.bfloat16)

    gmm_count = [0]

    def _gmm_effect(x, weight, group_list, config, **kwargs):
        call_index = gmm_count[0]
        gmm_count[0] += 1
        out_dim = _EP_INTER * 2 if call_index == 0 else _EP_HIDDEN
        return torch.zeros(x.shape[0], out_dim, dtype=torch.bfloat16)

    with ExitStack() as stack:
        stack.enter_context(patch(f'{_DISPATCHER}.dispatch_preprocess', return_value=(indices, split_sizes)))
        stack.enter_context(patch(f'{_DISPATCHER}.alltoall_dispatch', return_value=(dispatched, unpermute)))
        combine_mock = stack.enter_context(patch(f'{_DISPATCHER}.alltoall_combine', return_value=combined))
        gmm_mock = stack.enter_context(patch(f'{_MX_GMM}.mx_quant_group_gemm', side_effect=_gmm_effect))
        yield combine_mock, gmm_mock


def _ep_call(module, fused=True, fixed_router=False):
    """Invoke ep_forward_fused or ep_forward_eager with dummy inputs."""
    tokens = torch.zeros(_EP_TOKENS, _EP_HIDDEN, dtype=torch.bfloat16)
    index = torch.zeros(_EP_TOKENS, dtype=torch.int32)
    weights = torch.ones(_EP_TOKENS, dtype=torch.float32)
    fn = module.ep_forward_fused if fused else module.ep_forward_eager
    return fn(tokens, index, weights, ep_group=Mock(), fixed_router=fixed_router)


# -- delegation to _ep_forward -----------------------------------------------


def test_ep_forward_fused_delegates_with_fused_true():
    """ep_forward_fused must call _ep_forward(..., fused=True)."""
    module = _make_ep_module()
    module._ep_forward = Mock(return_value=torch.zeros(_EP_TOKENS, _EP_HIDDEN, dtype=torch.bfloat16))
    _ep_call(module, fused=True)
    assert module._ep_forward.call_args.kwargs['fused'] is True


def test_ep_forward_eager_delegates_with_fused_false():
    """ep_forward_eager must call _ep_forward(..., fused=False)."""
    module = _make_ep_module()
    module._ep_forward = Mock(return_value=torch.zeros(_EP_TOKENS, _EP_HIDDEN, dtype=torch.bfloat16))
    _ep_call(module, fused=False)
    assert module._ep_forward.call_args.kwargs['fused'] is False


# -- weight owner / attr wiring ----------------------------------------------


def test_ep_forward_propagates_weight_owner_and_attr():
    """Both GMMs in _ep_forward must carry the module as weight_owner
    and use the canonical attribute names 'gate_up_proj' / 'down_proj'.
    """
    module = _make_ep_module()
    with _ep_stubs() as (_combine, gmm):
        _ep_call(module)

    assert gmm.call_count == 2
    fc1, fc2 = gmm.call_args_list
    assert fc1.kwargs['weight_owner'] is module
    assert fc1.kwargs['weight_attr'] == 'gate_up_proj'
    assert fc2.kwargs['weight_owner'] is module
    assert fc2.kwargs['weight_attr'] == 'down_proj'


def test_ep_forward_passes_local_shard_not_dtensor_to_gmm():
    """_ep_forward must call _get_local_weight_ep on each parameter so that the
    local DTensor shard (not the global DTensor) is passed to mx_quant_group_gemm.

    With EP, parameters are distributed as DTensor (global shape [E, ...]).
    Passing the global DTensor as the weight arg causes MxGroupedMatmulBackward
    to return a local-shaped gradient [E/ep, ...] that is incompatible with the
    DTensor's global shape [E, ...], crashing backward. _get_local_weight_ep strips
    the DTensor; weight_owner/weight_attr handle gradient accumulation instead.
    """
    module = _make_ep_module()

    sentinel_gate_up = torch.zeros(_EP_EXPERTS * _EP_HIDDEN, _EP_INTER * 2, dtype=torch.bfloat16)
    sentinel_down = torch.zeros(_EP_EXPERTS * _EP_INTER, _EP_HIDDEN, dtype=torch.bfloat16)

    call_count = [0]

    def mock_local_weight(param):
        call_index = call_count[0]
        call_count[0] += 1
        return sentinel_gate_up if call_index == 0 else sentinel_down

    with _ep_stubs() as (_combine, gmm):
        with patch(
            f'{_DISPATCHER_MX}.swiglu',
            return_value=torch.zeros(_EP_TOKENS, _EP_INTER, dtype=torch.bfloat16),
        ):
            with patch.object(module, '_get_local_weight_ep', side_effect=mock_local_weight):
                _ep_call(module)

    fc1, fc2 = gmm.call_args_list
    assert fc1.kwargs['weight'] is sentinel_gate_up, (
        "gate_up_proj must be replaced by its local shard (not the DTensor) before calling mx_quant_group_gemm"
    )
    assert fc2.kwargs['weight'] is sentinel_down, (
        "down_proj must be replaced by its local shard (not the DTensor) before calling mx_quant_group_gemm"
    )


# -- activation dispatch -----------------------------------------------------


def test_ep_forward_situ_applies_act_fn_to_full_gate_up_tensor():
    """With _is_situ=True, _apply_activation must pass the full
    [tokens, 2*intermediate] gate-up tensor to act_fn, not a gate-only split.
    swiglu must not be called.
    """
    module = _make_ep_module(is_situ=True)
    with _ep_stubs() as (_combine, _gmm):
        with patch(f'{_DISPATCHER_MX}.swiglu') as swiglu_mock:
            _ep_call(module)

    assert module.act_fn.call_count == 1
    act_input = module.act_fn.call_args.args[0]
    assert act_input.shape[-1] == _EP_INTER * 2, (
        f"act_fn received shape {act_input.shape}; expected last dim {_EP_INTER * 2} (full gate+up)"
    )
    swiglu_mock.assert_not_called()


def test_ep_forward_non_situ_uses_swiglu():
    """With _is_situ=False, _apply_activation must call swiglu (standard SiGLU)
    and not invoke any act_fn.
    """
    module = _make_ep_module(is_situ=False)
    with _ep_stubs() as (_combine, _gmm):
        with patch(
            f'{_DISPATCHER_MX}.swiglu',
            return_value=torch.zeros(_EP_TOKENS, _EP_INTER, dtype=torch.bfloat16),
        ) as swiglu_mock:
            _ep_call(module)

    swiglu_mock.assert_called_once()
    assert module.act_fn is None


# -- zero-token guard --------------------------------------------------------


def test_ep_forward_zero_token_skips_gmm_and_preserves_output_shape():
    """When alltoall_dispatch returns an empty tensor (no tokens on this EP rank),
    mx_quant_group_gemm must not be called, and the final output must have the
    same shape as the input.
    """
    module = _make_ep_module()
    with _ep_stubs(tokens_per_expert=[0, 0]) as (_combine, gmm):
        with patch(
            f'{_DISPATCHER_MX}.swiglu',
            return_value=torch.zeros(0, _EP_INTER, dtype=torch.bfloat16),
        ):
            result = _ep_call(module)

    gmm.assert_not_called()
    assert result.shape == (_EP_TOKENS, _EP_HIDDEN)


# -- fused -> eager fallback -------------------------------------------------


def test_ep_forward_fused_falls_back_to_eager_when_any_expert_has_zero_tokens():
    """When any local expert receives zero tokens, _ep_forward must reset fused
    to False before calling alltoall_dispatch and alltoall_combine, because the
    fused permute kernel requires every expert to have at least one token.
    """
    module = _make_ep_module()

    # Expert 0 gets 2 tokens, expert 1 gets 0 — triggers the .any() fallback.
    expert_token_counts = torch.tensor([2, 0], dtype=torch.int64)
    indices = (expert_token_counts, torch.zeros(2, dtype=torch.int32))
    split_sizes = (
        torch.ones(_EP_EXPERTS, dtype=torch.int64),
        torch.ones(_EP_EXPERTS, dtype=torch.int64),
    )
    dispatched = torch.zeros(2, _EP_HIDDEN, dtype=torch.bfloat16)
    unpermute = (torch.arange(2), torch.arange(2))
    fused_seen = {}

    def capture_dispatch(ep_group, hidden_states, top_k_index, local_indices, split_sizes_arg, fused):
        fused_seen['dispatch'] = fused
        return dispatched, unpermute

    def capture_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes_arg, fused):
        fused_seen['combine'] = fused
        return torch.zeros(_EP_TOKENS, _EP_HIDDEN, dtype=torch.bfloat16)

    gmm_count = [0]

    def gmm_fn(x, weight, group_list, config, **kwargs):
        call_index = gmm_count[0]
        gmm_count[0] += 1
        return torch.zeros(x.shape[0], _EP_INTER * 2 if call_index == 0 else _EP_HIDDEN, dtype=torch.bfloat16)

    with ExitStack() as stack:
        stack.enter_context(patch(f'{_DISPATCHER}.dispatch_preprocess', return_value=(indices, split_sizes)))
        stack.enter_context(patch(f'{_DISPATCHER}.alltoall_dispatch', side_effect=capture_dispatch))
        stack.enter_context(patch(f'{_DISPATCHER}.alltoall_combine', side_effect=capture_combine))
        stack.enter_context(patch(f'{_MX_GMM}.mx_quant_group_gemm', side_effect=gmm_fn))
        _ep_call(module, fused=True)

    assert fused_seen['dispatch'] is False, (
        "alltoall_dispatch should receive fused=False when any expert has zero tokens"
    )
    assert fused_seen['combine'] is False
