#!/usr/bin/env python
"""Convert the full GLM-5.3-Flash checkpoint into a small training checkpoint.

Standalone CLI tool shipped next to train.py under examples/glm5.3_flash: it
depends on the repo's glm5_next modeling code (allowed here -- examples are a
user of fsdp_turbo), but train.py / run.sh do not depend on it. Run it
directly on any host with enough free RAM (the full model is materialized in
host memory). The directory name contains a dot, so module-style
``python -m`` invocation is impossible -- always run it as a plain script
(the bootstrap below makes the repo root importable from any cwd):

usage:
    python examples/glm5.3_flash/convert_checkpoint.py \
        --src /path/to/GLM-5.3-Flash --dst /path/to/out --num-hidden-layers 24

What this script does
----------------------
1. Loads the full fp8 (FinegrainedFP8) checkpoint and dequantizes it to pure
   bf16. The checkpoint's quantization_config carries no ``quant_method``, so
   transformers never instantiates a quantizer and hands over the raw fp8
   weights with their 128x128 block scales kept as sibling tensors; the
   scale-aware in-place dequantization from ``glm5_fp8_dequant`` (the exact
   path train.py uses) then converts everything to bf16.
   ``FinegrainedFP8HfQuantizer.validate_environment`` is still monkeypatched so
   that a transformers version which DOES instantiate the quantizer does not
   reject NPU/CPU hosts. NOTE: do NOT clear ``quantization_config`` in
   config.json to work around the check -- that loads the raw fp8 tensors as
   bf16 garbage.
2. Truncates the language model to ``--num-hidden-layers`` layers, slicing
   ``layer_types`` / ``mlp_layer_types`` / ``indexer_types`` to match.
   The vision tower and multimodal fusion are kept intact.
3. Slices routed experts down to ``--n-routed-experts``: expert
   gate_up/down weights, router weight and ``e_score_correction_bias`` all
   take the first N experts.
4. Saves a self-contained HF checkpoint:
   - ``quantization_config`` is scrubbed from the saved config.json (the saved
     weights are pure bf16; leaving it behind re-triggers the fp8 quantizer
     on the training host),
   - generation_config drops ``top_p`` / ``top_k`` / ``temperature`` with
     ``do_sample=False`` (transformers validation rejects the combination),
   - tokenizer/processor files are copied from the source directory.

Usage (on the training host; the FULL model is loaded into host RAM and
fp8 dequantized on CPU, so expect a long run -- that is the safe path)
--------------------------------------------------------------------
Pass everything on the command line:

    python examples/glm5.3_flash/convert_checkpoint.py \
        --src /path/to/GLM-5.3-Flash \
        --dst /path/to/glm53-flash-4l-64e \
        --num-hidden-layers 4 --n-routed-experts 64

The layer/expert targets carry NO default: always pass
``--num-hidden-layers`` / ``--n-routed-experts`` explicitly.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import time
import pathlib

# Make the repo root importable when this file is run as a plain script from
# any working directory: the local fsdp_turbo -- and with it the local
# glm5_next modeling file -- must win over any stale site-packages copy.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Conversion settings -- SRC/DST have placeholder defaults (override via CLI);
# the layer/expert reduction targets have NO default: pass them on the command
# line.
DEFAULT_SRC = "/path/to/GLM-5.3-Flash"
DEFAULT_DST = "/path/to/GLM-5.3-Flash-BF16"

import torch  # noqa: E402


def _print(msg):
    # Plain print(flush=True): the project logger setup gives __main__ no
    # handler, and conversion runs are piped to files on the training host.
    print(f"[convert] {msg}", flush=True)


def patch_finegrained_fp8_quantizer(src=None):
    """Neuter FineGrainedFP8HfQuantizer.validate_environment (any backend).

    If the source carries a quantization_config and no patch location
    matches, abort: loading would then dequantize incorrectly (or not at
    all) and silently bake garbage into the saved bf16 weights.
    """
    candidates = [
        # Known-good names on the training hosts (see train.py, which uses
        # exactly this module/class pair): capital-G class, "quantizer_" prefix.
        ("transformers.quantizers.quantizer_finegrained_fp8", "FineGrainedFP8HfQuantizer"),
        # Other transformers versions spell it differently; try them all --
        # a silently unmatched patch is what produced garbage weights before.
        ("transformers.quantizers.quantizer_finegrained_fp8", "FinegrainedFP8HfQuantizer"),
        ("transformers.quantizers.finegrained_fp8", "FineGrainedFP8HfQuantizer"),
        ("transformers.quantizers.finegrained_fp8", "FinegrainedFP8HfQuantizer"),
        ("transformers.quantization.quantizer.finegrained_fp8", "FinegrainedFP8HfQuantizer"),
        ("transformers.quantizers", "FineGrainedFP8HfQuantizer"),
        ("transformers.quantizers", "FinegrainedFP8HfQuantizer"),
    ]
    for module_name, class_name in candidates:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(mod, class_name, None)
        if cls is None:
            continue
        cls.validate_environment = lambda self, *args, **kwargs: None
        _print(f"patched {module_name}.{class_name}.validate_environment")
        return
    if src is not None:
        cfg_path = os.path.join(src, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                has_quant = json.load(f).get("quantization_config") is not None
            if has_quant:
                raise RuntimeError(
                    "Source is fp8-quantized but this transformers version's "
                    "FineGrainedFP8HfQuantizer could not be found to patch. "
                    "Loading would NOT dequantize correctly (silent garbage "
                    "weights). Pin the transformers version used by train.py "
                    "(module: transformers.quantizers.quantizer_finegrained_fp8)."
                )
    _print("FineGrainedFP8HfQuantizer not found; source treated as non-fp8")


def verify_dequantized(model):
    """Fail loudly if anything is still fp8 / carries block scales."""
    float8_prefixes = ("torch.float8",)
    bad_dtype, bad_keys = [], []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if str(tensor.dtype).startswith(float8_prefixes):
            bad_dtype.append(name)
        if "scale_inv" in name or "weight_scale" in name:
            bad_keys.append(name)
    if bad_dtype or bad_keys:
        raise RuntimeError(
            "The model was NOT dequantized to bf16 at load time:\n"
            f"  float8 tensors: {bad_dtype[:5]}\n  scale tensors: {bad_keys[:5]}\n"
            "Saving now would bake fp8 garbage in as bf16 (exactly the "
            "invalid-weights failure mode seen before). Check that the source "
            "config.json still carries its quantization_config and that the "
            "validate_environment patch above actually applied."
        )
    # Smoke check: untrained-looking garbage from raw fp8 bytes read as bf16
    # shows up as extreme magnitudes in the first elements.
    for name, param in model.named_parameters():
        flat = param.detach().flatten()[:4096]
        if flat.numel() and not torch.isfinite(flat).all():
            raise RuntimeError(f"non-finite values in {name} right after load")
    _print("all weights are bf16/fp32 and finite -- dequantization OK")


def truncate_layers(model, keep):
    lm = model.model.language_model
    total = len(lm.layers)
    if keep > total:
        raise ValueError(f"--num-hidden-layers={keep} exceeds source layers={total}")
    if total > keep:
        lm.layers = lm.layers[:keep]

    # A 'shared' DSA indexer layer needs a preceding 'full' layer *within the
    # kept slice*; otherwise forward() raises at runtime.
    last_full = None
    for idx, layer in enumerate(lm.layers):
        if layer.block_type == "linear_attention":
            continue
        if getattr(layer.self_attn, "indexer", None) is None:
            if last_full is None:
                raise RuntimeError(
                    f"kept layer {idx} is a shared-indexer DSA layer but no earlier "
                    "kept layer runs a full indexer -- increase --num-hidden-layers "
                    "or pick a slice whose first full-attention layer has a 'full' indexer"
                )
        else:
            last_full = idx
    if total > keep:
        _print(f"layers truncated: {total} -> {keep}")
    else:
        _print(f"model built with {total} layer(s) (config was pre-truncated)")
    for idx, layer in enumerate(lm.layers):
        _print(
            f"  layer {idx}: {layer.block_type}, mlp={'sparse' if layer.mlp.__class__.__name__ == 'Glm5NextTextMoE' else 'dense'}, "
            f"indexer={'yes' if getattr(layer.self_attn, 'indexer', None) is not None else '-'}"
        )
    return lm


def slice_experts(lm, n_experts, keep_cfg):
    """Slice routed experts (weights, router, score bias) to the first N."""
    n_group = getattr(keep_cfg, "n_group", 1) or 1
    topk_group = getattr(keep_cfg, "topk_group", 1) or 1
    top_k = getattr(keep_cfg, "num_experts_per_tok", 1) or 1
    if n_experts % n_group != 0:
        raise ValueError(f"n_routed_experts={n_experts} must be divisible by n_group={n_group}")
    if topk_group > n_group:
        raise ValueError(f"topk_group={topk_group} cannot exceed n_group={n_group}")
    if top_k > topk_group * (n_experts // n_group):
        raise ValueError(
            f"num_experts_per_tok={top_k} cannot fit in topk_group({topk_group}) x "
            f"experts_per_group({n_experts // n_group})"
        )

    sliced = 0
    for layer in lm.layers:
        mlp = layer.mlp
        if mlp.__class__.__name__ != "Glm5NextTextMoE":
            continue
        experts, router = mlp.experts, mlp.gate
        if experts.gate_up_proj.shape[0] < n_experts:
            raise RuntimeError(f"source has {experts.gate_up_proj.shape[0]} experts < target {n_experts}")
        experts.gate_up_proj.data = experts.gate_up_proj.data[:n_experts].contiguous()
        experts.down_proj.data = experts.down_proj.data[:n_experts].contiguous()
        experts.num_experts = n_experts
        router.weight.data = router.weight.data[:n_experts].contiguous()
        router.num_experts = n_experts
        if getattr(router, "e_score_correction_bias", None) is not None:
            router.e_score_correction_bias = router.e_score_correction_bias[:n_experts].contiguous()
        sliced += 1
    _print(f"routed experts sliced to {n_experts} in {sliced} MoE layer(s)")


def scrub_saved_config(dst):
    """Drop quantization_config from the saved config.json (weights are bf16)."""
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    changed = False
    if cfg.get("quantization_config") is not None:
        del cfg["quantization_config"]
        changed = True
    if "text_config" in cfg and cfg["text_config"].get("quantization_config") is not None:
        del cfg["text_config"]["quantization_config"]
        changed = True
    if changed:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        _print("scrubbed quantization_config from saved config.json")


def copy_tokenizer_files(src, dst):
    names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "tokenizer.model",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "image_processing_glm5_next.json",
        "video_processing_glm5_next.json",
    ]
    copied = []
    for name in names:
        path = os.path.join(src, name)
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(dst, name))
            copied.append(name)
    _print(f"copied tokenizer/processor files: {copied or 'NONE (copy them manually!)'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=DEFAULT_SRC, help="full GLM-5.3-Flash checkpoint dir")
    parser.add_argument("--dst", default=DEFAULT_DST, help="output checkpoint dir")
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        required=True,
        help="language-model layers to keep; no default",
    )
    parser.add_argument(
        "--n-routed-experts",
        type=int,
        required=True,
        help="routed experts to keep; no default",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.src):
        raise FileNotFoundError(
            f"--src {args.src!r} does not exist. Edit DEFAULT_SRC in this file or pass --src on the command line."
        )

    patch_finegrained_fp8_quantizer(src=args.src)

    # Import the LOCAL modeling file directly: AutoModelForImageTextToText
    # silently resolves glm5_next to transformers' native implementation and
    # bypasses this repo's fixes (meta-device buffers, NPU attention path).
    from fsdp_turbo.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    # Plain load path: from_pretrained WITHOUT an explicit config kwarg. The
    # source quantization_config carries no quant_method, so transformers loads
    # the fp8 weights raw (block scales kept as sibling weight_scale_inv
    # tensors) instead of auto-dequantizing; the scale-aware in-place
    # dequantization below -- the exact path train.py uses -- converts them
    # with the 128x128 block scales. Never blind-cast first: .to(bf16) on a
    # fp8 param drops its scale and silently corrupts the weights. Slow (full
    # model in host RAM, CPU dequant takes a while), but correct. Truncation
    # happens after the weights are safely loaded and verified.
    _print(f"loading FULL model from {args.src} (slow, but the safe path)...")
    t0 = time.time()
    model = Glm5NextForConditionalGeneration.from_pretrained(args.src, torch_dtype=torch.bfloat16, device_map="cpu")
    from fsdp_turbo.models.glm5_next.glm5_fp8_dequant import dequantize_glm5_experts_fp8

    dequantized_modules = dequantize_glm5_experts_fp8(model)
    _print(f"scale-aware fp8 dequant applied on {dequantized_modules} modules (in {time.time() - t0:.1f}s)")
    verify_dequantized(model)

    text_cfg = model.config.text_config
    lm = truncate_layers(model, args.num_hidden_layers)
    slice_experts(lm, args.n_routed_experts, text_cfg)

    # Config must describe exactly what was kept.
    text_cfg.num_hidden_layers = args.num_hidden_layers
    for field in ("layer_types", "mlp_layer_types", "indexer_types"):
        values = getattr(text_cfg, field)
        if values is not None and len(values) > args.num_hidden_layers:
            setattr(text_cfg, field, list(values[: args.num_hidden_layers]))
    text_cfg.n_routed_experts = args.n_routed_experts
    model.num_experts = args.n_routed_experts

    # The weights are pure bf16 now; make the saved checkpoint a plain model.
    model.config.quantization_config = None
    if getattr(model, "hf_quantizer", None) is not None:
        model.hf_quantizer = None

    # transformers validation rejects do_sample=False together with sampling flags.
    gc = model.generation_config
    if gc is not None:
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None

    os.makedirs(args.dst, exist_ok=True)
    _print(f"saving to {args.dst}...")
    model.save_pretrained(args.dst)
    scrub_saved_config(args.dst)
    copy_tokenizer_files(args.src, args.dst)

    _print(
        f"done: {args.num_hidden_layers} layers, {args.n_routed_experts} experts, "
        "n_group/topk_group/num_experts_per_tok unchanged from source. "
        "Reminder: expert_parallel_size must divide "
        f"{args.n_routed_experts} in config.yaml."
    )


if __name__ == "__main__":
    main()
