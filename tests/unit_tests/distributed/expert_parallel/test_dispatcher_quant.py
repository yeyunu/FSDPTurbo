from unittest.mock import Mock, patch

import torch

from fsdp_turbo.distributed.expert_parallel.dispatcher import (
    experts_computation,
    get_experts_forward_fn,
)


def _inputs():
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    gate_up = torch.randn(4 * 64, 128, dtype=torch.bfloat16)
    down = torch.randn(4 * 128, 64, dtype=torch.bfloat16)
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
            "fsdp_turbo.distributed.expert_parallel.dispatcher.quant_grouped_gemm",
            side_effect=lambda *a, **k: torch.randn(a[0].shape[0], 128, dtype=torch.bfloat16),
        ) as quant_gemm,
        patch("fsdp_turbo.distributed.expert_parallel.dispatcher.grouped_matmul"),
    ):
        experts_computation(
            x,
            torch.tensor([2, 2, 2, 2]),
            gate_up,
            down,
            torch.nn.functional.silu,
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


def test_experts_forward_fn_hands_module_as_owner():
    """get_experts_forward_fn must pass the MoE module itself into
    dispatch_mlp_combine as expert_module so the backward live-weight
    resolution can reach it.
    """
    module = Mock()
    module.hidden_dim = 64
    module.num_global_experts = 4
    module.expert_ids_per_ep_rank = torch.tensor([0, 1, 2, 3])
    module.gate_up_proj = torch.randn(4 * 64, 128, dtype=torch.bfloat16)
    module.down_proj = torch.randn(4 * 128, 64, dtype=torch.bfloat16)
    module.act_fn = torch.nn.functional.silu
    module.limit = None

    x = torch.randn(4, 64, dtype=torch.bfloat16)
    experts_forward = get_experts_forward_fn(Mock(), fused=False)

    with (
        patch(
            "fsdp_turbo.distributed.expert_parallel.dispatcher.dispatch_mlp_combine",
            return_value=torch.zeros(4, 64, dtype=torch.bfloat16),
        ) as combine,
        patch(
            "fsdp_turbo.distributed.expert_parallel.dispatcher.normalize_expert_args",
            side_effect=lambda i, w: (i, w),
        ),
    ):
        out = experts_forward(module, x, torch.tensor([0, 1, 2, 3]), torch.ones(4, dtype=torch.float32))

    assert combine.call_count == 1
    assert combine.call_args.kwargs["expert_module"] is module
    assert out.shape == x.shape


def test_experts_computation_without_quant_config_uses_plain_gemm():
    """Without quantization, the plain grouped_matmul path runs and no owner
    is required.
    """
    x, gate_up, down = _inputs()

    with (
        patch("fsdp_turbo.distributed.expert_parallel.dispatcher.quant_grouped_gemm") as quant_gemm,
        patch(
            "fsdp_turbo.distributed.expert_parallel.dispatcher.grouped_matmul",
            side_effect=lambda *a, **k: torch.randn(a[0].shape[0], 128, dtype=torch.bfloat16),
        ) as plain_gemm,
    ):
        experts_computation(
            x,
            torch.tensor([2, 2, 2, 2]),
            gate_up,
            down,
            torch.nn.functional.silu,
            act_limit=None,
            fused=False,
        )

    quant_gemm.assert_not_called()
    assert plain_gemm.call_count == 2
