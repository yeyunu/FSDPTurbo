# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Single-card integration check for use_fused_partial_rotary.

Builds the real DeepSeek-V4.1-Flash Attention layer (layer 0, sliding-window
only) from the vendored config, runs prefill forward+backward with the switch
off and on, and compares loss, input gradients and peak device memory.

Run from the repository root, with -m so *this* tree is imported:

    python -m tests.unit_tests.npu_integration_rope_fusion --device 7

A full 8-card training smoke is the stronger check but needs all cards free;
this proves the config -> ModelArgs -> Attention -> operator wiring, which is
what the change touches.

The operator is a hard requirement: without it both arms silently take the
reference path and every tolerance passes on identical graphs, so a missing
operator fails the run instead of reporting a vacuous PASS.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

# The A5 8-card config: same shape family the fusion targets.
_MODEL_CONFIG = Path(__file__).resolve().parents[2] / "examples/deepseek_v41/model_config/config_8p_A5.json"


def build_layer(args, dev, seed: int):
    """A fresh layer with identical weights for both runs.

    The reference Linear leaves its weight as torch.empty, so the model builder
    relies on initialize_deepseek_v41_model to fill it in; skipping that leaves
    an all-zero layer whose loss carries no signal.
    """
    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import initialize_deepseek_v41_model
    from fsdp_turbo.models.deepseek.deepseek_v41.attention import Attention

    torch.manual_seed(seed)
    layer = Attention(0, args)
    initialize_deepseek_v41_model(layer)
    # freqs_cis is complex64; casting the module wholesale would discard its
    # imaginary part, so it is preserved across the dtype conversion.
    freqs = layer.freqs_cis.clone()
    layer = layer.to(dev).to(torch.bfloat16)
    layer.freqs_cis = freqs.to(dev)
    return layer


def run_once(args, dev, seqlen: int, seed: int = 1234):
    from fsdp_turbo.utils.device import IS_CANN_OPS_TRANSFORMER_AVAILABLE

    layer = build_layer(args, dev, seed=seed)
    # Never setattr the switch here: this test exists to prove it arrives through
    # build_deepseek_v41_model_args -> ModelArgs -> Attention. Setting it after
    # construction would report "engaged" even if Attention__init__ ignored the
    # args entirely, so a broken wiring would still look engaged.
    expected = bool(args.use_fused_partial_rotary) and IS_CANN_OPS_TRANSFORMER_AVAILABLE
    engaged = bool(layer.use_fused_partial_rotary)

    torch.manual_seed(7)
    x = torch.randn(1, seqlen, args.dim, dtype=torch.bfloat16, device=dev)
    x.requires_grad_(True)

    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    base = torch.npu.memory_allocated()

    out = layer(x, 0)
    loss = out.float().pow(2).mean()
    loss.backward()
    torch.npu.synchronize()

    peak = torch.npu.max_memory_allocated() - base
    return {
        "engaged": engaged,
        "expected": expected,
        "loss": loss.item(),
        "grad_norm": x.grad.float().norm().item(),
        "grad": x.grad.detach().float().cpu(),
        "peak_mib": peak / 2**20,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=7)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--grad-tol", type=float, default=0.05)
    args_ns = ap.parse_args()

    import torch_npu  # noqa: F401

    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import build_deepseek_v41_model_args
    from fsdp_turbo.utils.device import IS_CANN_OPS_TRANSFORMER_AVAILABLE

    torch.npu.set_device(args_ns.device)
    dev = torch.device(f"npu:{args_ns.device}")
    available = IS_CANN_OPS_TRANSFORMER_AVAILABLE
    print(f"device={dev} seq={args_ns.seq} operator_available={available}")
    if not available:
        # Hard gate.  Without the operator both arms fall back to the reference
        # path, so the graphs are identical and every tolerance below passes
        # vacuously -- a false success exactly where a regression matters most.
        print(
            "\n[FAIL] inplace_partial_rotary_mul is unavailable (cann_ops_transformer "
            "missing or too old): OFF and ON would both run the reference path, so "
            "this run would not cover the fused operator at all."
        )
        return 1

    base_args = build_deepseek_v41_model_args(config_path=_MODEL_CONFIG)
    print(
        f"model: dim={base_args.dim} heads={base_args.n_heads} "
        f"head_dim={base_args.head_dim} rope_dim={base_args.rope_head_dim} "
        f"window={base_args.window_size} n_layers={base_args.n_layers}"
    )

    off = run_once(base_args, dev, args_ns.seq)
    on_args = replace(base_args, use_fused_partial_rotary=True)
    on = run_once(on_args, dev, args_ns.seq)

    print("\n{:<10} {:>16} {:>16}".format("", "switch OFF", "switch ON"))
    print("{:<10} {:>16.8f} {:>16.8f}".format("loss", off["loss"], on["loss"]))
    print("{:<10} {:>16.4f} {:>16.4f}".format("|x.grad|", off["grad_norm"], on["grad_norm"]))
    print("{:<10} {:>16.1f} {:>16.1f}".format("peak MiB", off["peak_mib"], on["peak_mib"]))

    ok = True
    if not on["engaged"] or on["engaged"] != on["expected"]:
        print(
            "\n[FAIL] use_fused_partial_rotary did not reach the Attention layer "
            f"(args->layer expects {on['expected']}, layer has {on['engaged']})"
        )
        ok = False
    elif off["engaged"] or off["engaged"] != off["expected"]:
        print(
            f"\n[FAIL] the OFF arm is not on the reference path (expects {off['expected']}, layer has {off['engaged']})"
        )
        ok = False
    else:
        print(
            "\n[PASS] use_fused_partial_rotary reached the Attention layer, and "
            "the OFF arm stayed on the reference path"
        )

    denom = abs(off["loss"]) or 1.0
    loss_rel = abs(on["loss"] - off["loss"]) / denom
    p = loss_rel < 0.02
    ok &= p
    print(f"[{'PASS' if p else 'FAIL'}] loss rel diff = {loss_rel:.6f} (<0.02, bf16 cos/sin explains ~0.6%)")

    gd = (on["grad"] - off["grad"]).abs().max().item()
    gs = off["grad"].abs().max().item() or 1.0
    grad_rel = gd / gs
    p = grad_rel < args_ns.grad_tol
    ok &= p
    print(f"[{'PASS' if p else 'FAIL'}] d(loss)/dx rel diff = {grad_rel:.6f} (<{args_ns.grad_tol})")

    saved = off["peak_mib"] - on["peak_mib"]
    print(f"[info] peak memory {off['peak_mib']:.1f} -> {on['peak_mib']:.1f} MiB (saved {saved:.1f} MiB)")
    if abs(saved) < 1.0:
        print(
            "       no difference at this sequence length: the operator's fixed "
            "16 MiB workspace only beats the reference temporaries once the q/o "
            "slices are large (S=4096 measures 128 -> 16 MiB per call)."
        )
        print("       The memory win is quantified in the feasibility report; this test covers wiring and numerics.")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
