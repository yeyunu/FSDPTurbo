# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# This file is adapted from the DeepSeek-V4 reference model implementation
# (DeepSeek-AI official release) for FSDPTurbo training: dense-attention and
# FSDPTurbo adaptations (expert-parallel dispatch, reduced config, hash-MoE
# buffers).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at LICENSE (Apache-2.0) in the repo root.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load a matching subset of the real DeepSeek-V4-Flash-BF16 weights into the
stock transformers (>=5.8.1) deepseek_v4 model. Works on CPU/NPU/GPU.

The released checkpoints ships novamlx-style key names; transformers uses the
standard HF layout, so the translate is not a plain prefix rewrite:

  * ``embed.weight`` / ``head.weight`` / ``norm.weight`` (top level) map onto
    ``model.embed_tokens`` / ``lm_head`` / ``model.norm``.
  * ``layers.N.attn.*`` -> ``model.layers.N.self_attn.*`` with the component
    renames ``wq_a/wq_b/wkv/wo_a/wo_b`` -> ``q_a_proj/q_b_proj/kv_proj/
    o_a_proj/o_b_proj``, ``q_norm`` -> ``q_a_norm``, ``attn_sink`` ->
    ``sinks``.
  * compressor/indexer ``ape`` -> ``position_bias``, ``wgate`` ->
    ``gate_proj``, ``wkv`` -> ``kv_proj``, ``norm`` -> ``kv_norm``;
    ``indexer.weights_proj`` -> ``indexer.scorer.weights_proj``,
    ``indexer.wq_b`` -> ``indexer.q_b_proj``.
  * ``attn_norm`` / ``ffn_norm`` -> ``input_layernorm`` /
    ``post_attention_layernorm``; ``hc_attn_*`` / ``hc_ffn_*`` ->
    ``attn_hc.*`` / ``ffn_hc.*``; top-level ``hc_head_*`` ->
    ``model.hc_head.hc_*``.
  * ``ffn.gate.bias`` -> ``mlp.gate.e_score_correction_bias`` (a buffer);
    ``ffn.gate.tid2eid`` maps straight through.
  * ``ffn.shared_experts.w1/w2/w3`` -> ``mlp.shared_experts.gate_proj/
    down_proj/up_proj``.
  * ``ffn.experts.N.w1 + w3`` pack into ``mlp.experts.gate_up_proj[N]`` and
    ``ffn.experts.N.w2`` -> ``mlp.experts.down_proj[N]`` (expert rows beyond
    the reduced 16-expert count are skipped).
  * ``mtp.*`` (MTP blocks of the released checkpoint) are always skipped.

Unknown / shape-mismatched keys are reported in the returned stats.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from pathlib import Path

import torch
from safetensors import safe_open

from fsdp_turbo.models.deepseek.deepseek_v4.adapter import (
    build_deepseek_v4_model,
    build_deepseek_v4_model_args,
)
from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)

# novamlx component -> HF component renames, applied inside layers.N.attn.*
_ATTN_RENAMES = {
    "wq_a.weight": "q_a_proj.weight",
    "q_norm.weight": "q_a_norm.weight",
    "wq_b.weight": "q_b_proj.weight",
    "wkv.weight": "kv_proj.weight",
    "kv_norm.weight": "kv_norm.weight",
    "wo_a.weight": "o_a_proj.weight",
    "wo_b.weight": "o_b_proj.weight",
    "attn_sink": "sinks",
}
_COMPRESSOR_RENAMES = {
    "ape": "position_bias",
    "wgate.weight": "gate_proj.weight",
    "wkv.weight": "kv_proj.weight",
    "norm.weight": "kv_norm.weight",
}
_TOP_LEVEL_RENAMES = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_scale": "model.hc_head.hc_scale",
}
_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
_GATE_RE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.(weight|bias)$")
_LAYER_NORM_RE = re.compile(r"^layers\.(\d+)\.(attn_norm|ffn_norm)\.weight$")
_HC_RE = re.compile(r"^layers\.(\d+)\.(hc_attn|hc_ffn)_(fn|base|scale)$")
_SHARED_RE = re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w([123])\.weight$")
_INDEXER_RE = re.compile(r"^layers\.(\d+)\.attn\.indexer\.compressor\.(ape|wgate\.weight|wkv\.weight|norm\.weight)$")


def _translate_key(real_key: str) -> str | None:
    """Map a released (novamlx) key to the transformers key, or ``None`` to skip."""
    if real_key.startswith("mtp."):
        return None
    if real_key in _TOP_LEVEL_RENAMES:
        return _TOP_LEVEL_RENAMES[real_key]
    m = _LAYER_NORM_RE.match(real_key)
    if m:
        which = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}[m.group(2)]
        return f"model.layers.{m.group(1)}.{which}.weight"
    m = _HC_RE.match(real_key)
    if m:
        site = {"hc_attn": "attn_hc", "hc_ffn": "ffn_hc"}[m.group(2)]
        return f"model.layers.{m.group(1)}.{site}.{m.group(3)}"
    if real_key.startswith("layers.") and ".attn.compressor." in real_key:
        pre = real_key.split(".attn.compressor.", 1)[0]  # layers.N.attn.compressor|indexer.compressor
        layer = int(pre.split(".")[1])
        if ".indexer.compressor." in real_key:
            # layers.N.attn.indexer.compressor.X  -> self_attn.indexer.X
            _, suffix = real_key.split(".indexer.compressor.", 1)
            renamed = _COMPRESSOR_RENAMES.get(suffix)
            if renamed is None:
                return None
            return f"model.layers.{layer}.self_attn.indexer.{renamed}"
        suffix = real_key.split(".attn.compressor.", 1)[1]
        renamed = _COMPRESSOR_RENAMES.get(suffix)
        if renamed is None:
            return None
        return f"model.layers.{layer}.self_attn.compressor.{renamed}"
    if ".attn.indexer." in real_key:
        layer = int(real_key.split(".")[1])
        if real_key.endswith("indexer.weights_proj.weight"):
            return f"model.layers.{layer}.self_attn.indexer.scorer.weights_proj.weight"
        if real_key.endswith("indexer.wq_b.weight"):
            return f"model.layers.{layer}.self_attn.indexer.q_b_proj.weight"
        return None
    if real_key.startswith("layers.") and ".attn." in real_key:
        layer = int(real_key.split(".")[1])
        head, _, tail = real_key.partition(".attn.")
        if not tail:
            return None
        renamed = _ATTN_RENAMES.get(tail)
        if renamed is None:
            return None
        return f"model.layers.{layer}.self_attn.{renamed}"
    if real_key.endswith(".ffn.gate.tid2eid"):
        layer = int(real_key.split(".")[1])
        return f"model.layers.{layer}.mlp.gate.tid2eid"
    return None


def load_weights_into(model: torch.nn.Module, weights_dir: str, device: str = "cpu") -> dict:
    """Load matching real weights into ``model`` (transformers layout). Returns a stats dict."""
    weights_dir = Path(weights_dir)
    index_path = weights_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({weights_dir / fn for fn in set(weight_map.values())})
    else:
        shards = sorted(glob.glob(str(weights_dir / "*.safetensors")))
        weight_map = {}
    real_keys: set[str] = set()
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            real_keys.update(sf.keys())

    target_keys = set(model.state_dict().keys())
    stats = {
        "loaded": 0,
        "skipped_shape_mismatch": [],
        "skipped_extra": [],
    }

    state = model.state_dict()
    direct_state = {}
    gate_crop = {}
    expert_pack: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    for real_key in sorted(real_keys):
        if real_key.startswith("mtp."):
            continue
        m = _GATE_RE.match(real_key)
        if m:
            layer, which = int(m.group(1)), m.group(2)
            target = f"model.layers.{layer}.mlp.gate.{'weight' if which == 'weight' else 'e_score_correction_bias'}"
            if target not in target_keys:
                stats["skipped_extra"].append(real_key)
                continue
            gate_crop[real_key] = target
            continue
        m = _EXPERT_RE.match(real_key)
        if m:
            layer, exp, which = int(m.group(1)), int(m.group(2)), m.group(3)
            if exp >= state["model.layers.0.mlp.experts.gate_up_proj"].shape[0]:
                stats["skipped_extra"].append(f"experts layer {layer} index {exp} (>= num_experts)")
                continue
            expert_pack.setdefault((layer, exp), {})[which] = real_key
            continue
        m = _SHARED_RE.match(real_key)
        if m:
            layer, which = int(m.group(1)), m.group(2)
            proj = {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[which]
            target = f"model.layers.{layer}.mlp.shared_experts.{proj}.weight"
            if target not in target_keys:
                stats["skipped_extra"].append(real_key)
                continue
            direct_state[real_key] = target
            continue
        target = _translate_key(real_key)
        if target is None:
            stats["skipped_extra"].append(real_key)
            continue
        if target not in target_keys:
            stats["skipped_extra"].append(real_key)
            continue
        direct_state[real_key] = target

    # second pass: read tensors per shard and apply
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            for real_key in sf.keys():
                if real_key not in direct_state and real_key not in gate_crop:
                    continue
                tensor = sf.get_tensor(real_key)
                if real_key in gate_crop:
                    target = gate_crop[real_key]
                    tgt = state[target]
                    if tuple(tensor.shape) == tuple(tgt.shape):
                        state[target] = tensor.to(device=device, dtype=tgt.dtype)
                        stats["loaded"] += 1
                    elif tensor.dim() == tgt.dim() and all(a >= b for a, b in zip(tensor.shape, tgt.shape)):
                        sl = tuple(slice(0, b) for b in tgt.shape)
                        state[target] = tensor[sl].to(device=device, dtype=tgt.dtype)
                        stats["loaded"] += 1
                    else:
                        stats["skipped_shape_mismatch"].append((real_key, tuple(tensor.shape), tuple(tgt.shape)))
                    continue
                target = direct_state[real_key]
                tgt = state[target]
                if tuple(tensor.shape) == tuple(tgt.shape):
                    state[target] = tensor.to(device=device, dtype=tgt.dtype)
                    stats["loaded"] += 1
                else:
                    stats["skipped_shape_mismatch"].append((real_key, tuple(tensor.shape), tuple(tgt.shape)))

    # expert packing
    gu_shape = state["model.layers.0.mlp.experts.gate_up_proj"].shape
    dp_shape = state["model.layers.0.mlp.experts.down_proj"].shape
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            for (layer, exp), which_map in expert_pack.items():
                for which, real_key in which_map.items():
                    if real_key in sf.keys():
                        which_map[which] = sf.get_tensor(real_key)  # replace key with tensor
    for (layer, exp), tensors in expert_pack.items():
        gu_key = f"model.layers.{layer}.mlp.experts.gate_up_proj"
        dp_key = f"model.layers.{layer}.mlp.experts.down_proj"
        if gu_key not in target_keys or dp_key not in target_keys:
            continue
        w1, w3, w2 = tensors.get("w1"), tensors.get("w3"), tensors.get("w2")
        if w1 is None or w3 is None or w2 is None:
            continue
        gu = state[gu_key]  # [E, 2*I, H]
        dp = state[dp_key]  # [E, H, I]
        if tuple(w1.shape) == tuple(w3.shape) and w1.shape[0] * 2 == gu_shape[1] and w1.shape[1] == gu_shape[2]:
            gu[exp] = torch.cat([w1, w3], dim=0).to(device=device, dtype=gu.dtype)
            stats["loaded"] += 2
        else:
            stats["skipped_shape_mismatch"].append((f"gate_up_proj L{layer}", tuple(w1.shape), gu_shape))
        if tuple(w2.shape) == tuple(dp_shape[1:]):
            dp[exp] = w2.to(device=device, dtype=dp.dtype)
            stats["loaded"] += 1
        else:
            stats["skipped_shape_mismatch"].append((f"down_proj L{layer}", tuple(w2.shape), dp_shape))

    # IMPORTANT: assigning into model.state_dict() does not write back into the
    # module. Reapply the collected state via load_state_dict.
    model.load_state_dict(state, strict=False)
    stats["skipped_missing_in_real_sample"] = []
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="Path to the released DeepSeek-V4-Flash-BF16 safetensors dir")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not args.weights:
        ap.error("--weights is required (e.g. --weights /path/to/DeepSeek-V4-Flash-BF16)")

    model_args = build_deepseek_v4_model_args()
    print_rank(
        logger.info,
        "MODEL_ARGS: "
        + str(
            dict(
                n_layers=model_args.num_hidden_layers,
                n_routed=model_args.n_routed_experts,
                n_mtp=model_args.num_nextn_predict_layers,
                n_hash=int(sum(1 for t in model_args.mlp_layer_types if t == "hash_moe")),
            )
        ),
    )
    model = build_deepseek_v4_model(model_args=model_args)

    stats = load_weights_into(model, args.weights, device=args.device)
    print_rank(logger.info, "LOAD STATS COMPACT:")
    print_rank(logger.info, f"  loaded keys: {stats['loaded']}")
    print_rank(logger.info, f"  shape mismatches: {len(stats['skipped_shape_mismatch'])}")
    for x in stats["skipped_shape_mismatch"][:10]:
        print_rank(logger.info, f"    SHAPE: {x}")
    print_rank(logger.info, f"  unmatched real keys (skipped): {len(stats['skipped_extra'])}")
    print_rank(logger.info, "DONE-LOAD")


if __name__ == "__main__":
    main()
