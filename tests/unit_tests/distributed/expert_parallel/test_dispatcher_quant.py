from unittest.mock import Mock, patch

import pytest
import torch

from fsdp_turbo.distributed.expert_parallel.dispatcher import (
    experts_computation,
)
from fsdp_turbo.distributed.expert_parallel.expert_parallel import get_experts_forward_fn
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig
from fsdp_turbo.ops.moe import grouped_matmul


def _inputs():
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    gate_up = torch.randn(4, 128, 64, dtype=torch.bfloat16)
    down = torch.randn(4, 64, 64, dtype=torch.bfloat16)
    return x, gate_up, down


def test_experts_computation_passes_owner_and_attr_to_quant_gemm():
    """The MXFP8 path must forward the expert module and canonical attribute
    names into quant_grouped_gemm so backward can resolve live weights.
    """
    x, gate_up, down = _inputs()
    quant_config = Mock()
    quant_config.get_key_dtype.return_value = torch.bfloat16
    owner = Mock()

    with (
        patch(
            "fsdp_turbo.quantization.utils.quant_grouped_gemm",
            side_effect=lambda *a, **k: torch.randn(a[0].shape[0], a[1].shape[-2], dtype=torch.bfloat16),
        ) as quant_gemm,
        patch("fsdp_turbo.distributed.expert_parallel.dispatcher.grouped_matmul"),
    ):
        experts_computation(
            x,
            torch.tensor([2, 2, 2, 2]),
            gate_up,
            down,
            act_fn=None,
            act_limit=None,
            fused=False,
            quant_config=quant_config,
            weight_owner=owner,
        )

    assert quant_gemm.call_count == 2
    fc1 = quant_gemm.call_args_list[0]
    fc2 = quant_gemm.call_args_list[1]
    assert fc1.args[1] is gate_up
    assert fc1.kwargs["weight_owner"] is owner
    assert fc1.kwargs["weight_attr"] == "gate_up_proj"
    assert fc2.args[1] is down
    assert fc2.kwargs["weight_owner"] is owner
    assert fc2.kwargs["weight_attr"] == "down_proj"


@pytest.mark.parametrize(
    "dispatcher, dispatch_name",
    [
        ("all2all", "all2all_experts_forward_fn"),
        ("allgather", "all_gather_experts_forward_fn"),
        ("chunkmoe", "chunkmoe_experts_forward_fn"),
        ("mc2", "mc2_experts_forward_fn"),
    ],
)
@pytest.mark.parametrize("use_fused_ops", [False, True])
def test_experts_forward_fn_hands_module_as_owner(dispatcher, dispatch_name, use_fused_ops):
    """Each dispatcher receives the expert owner and its quantization config."""
    module = Mock()
    module.hidden_dim = 64
    module.num_global_experts = 4
    module.expert_ids_per_ep_rank = torch.tensor([0, 1, 2, 3])
    module.gate_up_proj = torch.randn(4, 128, 64, dtype=torch.bfloat16)
    module.down_proj = torch.randn(4, 64, 64, dtype=torch.bfloat16)
    module.act_fn = torch.nn.functional.silu
    module.limit = None
    module.alpha = None
    module._quant_config = Mock()
    ep_group = Mock()
    ep_plan = EPPlanConfig(dispatcher=dispatcher, use_fused_ops=use_fused_ops, use_swiglu=False)
    hidden_states = torch.randn(4, 64, dtype=torch.bfloat16)
    expert_indices = torch.tensor([[0], [1], [2], [3]])
    expert_weights = torch.ones(4, 1, dtype=torch.float32)

    with patch(
        f"fsdp_turbo.distributed.expert_parallel.expert_parallel.{dispatch_name}",
        return_value=torch.zeros_like(hidden_states),
    ) as dispatch:
        experts_forward = get_experts_forward_fn(ep_group, ep_plan, dispatcher=dispatcher)
        out = experts_forward(module, hidden_states, expert_indices, expert_weights)

    dispatch.assert_called_once()
    assert dispatch.call_args.args[6] is ep_group
    assert dispatch.call_args.kwargs["expert_module"] is module
    assert dispatch.call_args.kwargs["quant_config"] is module._quant_config
    assert dispatch.call_args.kwargs["ep_plan"] is ep_plan
    assert dispatch.call_args.kwargs["act_fn"] is module.act_fn
    assert dispatch.call_args.kwargs["fused"] is use_fused_ops
    assert out.shape == hidden_states.shape


def test_experts_computation_without_quant_config_uses_plain_gemm():
    """Without quantization, the plain grouped_matmul path runs and no owner
    is required.
    """
    x, gate_up, down = _inputs()

    with (
        patch("fsdp_turbo.quantization.utils.quant_grouped_gemm") as quant_gemm,
        patch(
            "fsdp_turbo.distributed.expert_parallel.dispatcher.grouped_matmul",
            wraps=grouped_matmul,
        ) as plain_gemm,
    ):
        output = experts_computation(
            x,
            torch.tensor([2, 2, 2, 2]),
            gate_up,
            down,
            act_fn=None,
            act_limit=None,
            fused=False,
        )

    quant_gemm.assert_not_called()
    assert plain_gemm.call_count == 2
    assert output.shape == x.shape


@pytest.mark.parametrize("dispatcher", ["eager", "fused", "alltoall", "unknown", None, 42])
def test_unsupported_dispatcher_fails_before_forward(dispatcher):
    """Direct factory calls must not return a partial with dispatch_fn=None."""
    with pytest.raises(RuntimeError, match="Unsupported dispatcher"):
        get_experts_forward_fn(Mock(), EPPlanConfig(), dispatcher=dispatcher)


def test_domino_dispatcher_retains_its_factory_interface():
    ep_group = Mock()
    plan = EPPlanConfig(dispatcher="domino", fixed_router=True)
    with patch(
        "fsdp_turbo.distributed.expert_parallel.expert_parallel.get_domino_experts_forward_fn"
    ) as domino_factory:
        forward = get_experts_forward_fn(ep_group, plan, dispatcher=plan.dispatcher)

    domino_factory.assert_called_once_with(ep_group, fixed_router=True)
    assert forward is domino_factory.return_value


def test_custom_dispatcher_preserves_module_method_arguments():
    ep_group = Mock()
    plan = EPPlanConfig(dispatcher="custom_ep_forward_eager", fixed_router=True)
    module = Mock()
    forward = get_experts_forward_fn(ep_group, plan, dispatcher=plan.dispatcher)
    hidden_states, indices, weights = Mock(), Mock(), Mock()

    output = forward(module, hidden_states, indices, weights)

    module.ep_forward_eager.assert_called_once_with(
        hidden_states, indices, weights, ep_group=ep_group, fixed_router=True
    )
    assert output is module.ep_forward_eager.return_value


def test_callable_dispatcher_preserves_group_first_arguments():
    ep_group = Mock()
    dispatcher = Mock()
    plan = EPPlanConfig(dispatcher=dispatcher)
    forward = get_experts_forward_fn(ep_group, plan, dispatcher=plan.dispatcher)
    hidden_states = Mock()

    output = forward(hidden_states)

    dispatcher.assert_called_once_with(ep_group, hidden_states)
    assert output is dispatcher.return_value
