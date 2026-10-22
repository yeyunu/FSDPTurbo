from pathlib import Path
from types import SimpleNamespace

import torch

from fsdp_turbo.models.deepseek.deepseek_v41.adapter import build_deepseek_v41_model_args
from fsdp_turbo.models.deepseek.deepseek_v41.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
    _match_checkpoint_layout,
)

_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/deepseek_v41/model_config"


def _adapter(num_experts=2):
    return DeepSeekV41StateDictAdapter(
        SimpleNamespace(
            n_routed_experts=num_experts,
            moe_inter_dim=3,
            vision_patch_size=2,
        )
    )


def test_full_pretrained_model_config():
    args = build_deepseek_v41_model_args(config_path=_MODEL_CONFIG_DIR / "config_full_bf16.json")

    assert args.n_layers == 40
    assert args.n_routed_experts == 384
    assert args.dtype == "bf16"
    assert args.expert_dtype is None
    assert args.n_mtp_layers == 0
    assert args.dspark_block_size == 0
    assert tuple(args.kv_source_layers) == (2, 8, 14, 20)
    assert tuple(args.engram_layer_ids) == (1, 14)


def test_official_flat_vision_patch_layout_is_accepted():
    patch = torch.empty(5, 3, 2, 2)

    matched = _match_checkpoint_layout(
        "vision.patch_embed.proj.weight",
        patch,
        (5, 12),
    )

    assert matched.shape == (5, 12)


def test_plain_expert_weights_round_trip():
    adapter = _adapter()
    gate_up = torch.arange(2 * 6 * 4).reshape(2, 6, 4)
    down = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    native = {
        "model.layers.0.ffn.experts.gate_up_proj": gate_up,
        "model.layers.0.ffn.experts.down_proj": down,
    }

    hf = adapter.to_hf(native)

    assert set(hf) == {
        f"layers.0.ffn.experts.{expert}.{projection}.weight" for expert in range(2) for projection in ("w1", "w2", "w3")
    }
    torch.testing.assert_close(hf["layers.0.ffn.experts.1.w1.weight"], gate_up[1, :3])
    torch.testing.assert_close(hf["layers.0.ffn.experts.1.w3.weight"], gate_up[1, 3:])
    torch.testing.assert_close(hf["layers.0.ffn.experts.1.w2.weight"], down[1])

    restored = adapter.from_hf(hf)
    torch.testing.assert_close(
        restored["model.layers.0.ffn.experts.gate_up_proj"],
        gate_up,
    )
    torch.testing.assert_close(
        restored["model.layers.0.ffn.experts.down_proj"],
        down,
    )


def test_direct_and_vision_patch_weights_round_trip():
    adapter = _adapter()
    patch = torch.arange(5 * 12).reshape(5, 12)
    native = {
        "model.embed.weight": torch.randn(7, 4),
        "model.layers.0.attn.wq_a.weight": torch.randn(3, 4),
        "model.vision.patch_embed.proj.weight": patch,
        "model.image_start": torch.randn(4),
    }

    hf = adapter.to_hf(native)

    assert set(hf) == {
        "embed.weight",
        "layers.0.attn.wq_a.weight",
        "vision.patch_embed.proj.weight",
        "image_start",
    }
    assert hf["vision.patch_embed.proj.weight"].shape == (5, 3, 2, 2)
    restored = adapter.from_hf(hf)
    assert set(restored) == set(native)
    for key, value in native.items():
        torch.testing.assert_close(restored[key], value)


def test_expected_hf_keys_expand_all_experts():
    adapter = _adapter(num_experts=3)
    native = {
        "model.norm.weight": torch.empty(4),
        "model.layers.2.ffn.experts.gate_up_proj": torch.empty(3, 6, 4),
        "model.layers.2.ffn.experts.down_proj": torch.empty(3, 4, 3),
    }

    expected = adapter.expected_hf_keys(native)

    assert "norm.weight" in expected
    assert len(expected) == 10
    for expert in range(3):
        for projection in ("w1", "w2", "w3"):
            assert f"layers.2.ffn.experts.{expert}.{projection}.weight" in expected


def test_incomplete_expert_group_fails():
    adapter = _adapter()
    hf = {
        "layers.0.ffn.experts.0.w1.weight": torch.empty(3, 4),
        "layers.0.ffn.experts.0.w3.weight": torch.empty(3, 4),
    }

    try:
        adapter.from_hf(hf)
    except ValueError as error:
        assert "Incomplete routed expert weights" in str(error)
    else:
        raise AssertionError("Incomplete expert weights must fail.")
