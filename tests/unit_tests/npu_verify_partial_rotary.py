# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Numeric verification for the fused partial-RoPE path.

Run on an Ascend950DT node from the repository root.  Use ``-m`` (not a bare
path): fsdp_turbo is editable-installed from another checkout, so only the
current directory on sys.path selects *this* tree.

    python -m tests.unit_tests.npu_verify_partial_rotary --device 7

Checks, at the real V4.1-Flash q/o shapes:
  1. forward equivalence against the reference complex implementation
     (unrotated channels bit-exact, rotated tail within bf16 cos/sin precision)
  2. backward equivalence against autograd through that reference
  3. the inverse (conjugate) path
  4. that the cos/sin cache cannot serve stale values when freqs_cis changes
  5. the disabled path still takes the reference implementation
"""

import argparse
import sys

import torch


# An *independent* reference: the complex maths the operator has to reproduce,
# kept here so the operator is not merely compared against the same tree that
# calls it.  It takes the trailing slice explicitly, which is the convention
# production apply_rotary_emb now hides internally, so check_production_drift()
# pins the two together -- if production changes convention again, that check
# fails instead of silently invalidating every numeric result below.
def apply_rotary_emb(x, freqs_cis, inverse=False):
    """Rotate `x` in place, taking adjacent element pairs as complex numbers."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def check_production_drift(x, freqs, rd, inverse=False):
    """Production ``apply_rotary_emb`` must still agree with the reference above.

    Production now delegates to ``ops.rope.inplace_rope`` and slices the rotary
    block internally, so both are exercised on the *full* tensor and the
    unrotated head is checked to stay untouched.
    """
    from fsdp_turbo.models.deepseek.deepseek_v41.layers import (
        apply_rotary_emb as production_apply_rotary_emb,
    )

    got = x.clone()
    production_apply_rotary_emb(got, freqs, inverse, use_triton_inplace_rope=False)
    ref = x.clone()
    apply_rotary_emb(ref[..., -rd:], freqs, inverse)
    drift = (got - ref).abs().max().item()
    head = (got[..., :-rd] - x[..., :-rd]).abs().max().item()
    return drift, head


def build_freqs(seqlen, rd, dev, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, rd, 2).float() / rd))
    f = torch.outer(torch.arange(seqlen).float(), inv)
    return torch.polar(torch.ones_like(f), f).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=7)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--rd", type=int, default=64)
    args = ap.parse_args()

    import torch_npu  # noqa: F401

    from fsdp_turbo.ops.npu.rope import _pack_partial_coefficients
    from fsdp_turbo.ops.rope import partial_rope
    from fsdp_turbo.utils.device import IS_CANN_OPS_TRANSFORMER_AVAILABLE

    torch.npu.set_device(args.device)
    dev = torch.device(f"npu:{args.device}")
    dt = torch.bfloat16
    s, h, D, rd = args.seq, args.heads, args.head_dim, args.rd
    freqs = build_freqs(s, rd, dev)
    print(f"device={dev} seq={s} heads={h} head_dim={D} rope_dim={rd}")
    print(f"fused operator available: {IS_CANN_OPS_TRANSFORMER_AVAILABLE}")
    if not IS_CANN_OPS_TRANSFORMER_AVAILABLE:
        print("FAIL: cann_ops_transformer / inplace_partial_rotary_mul missing")
        return 1

    ok = True

    def stage(msg):
        """Sync so a device fault is attributed to the stage that caused it."""
        torch.npu.synchronize()
        print(f"  .. {msg}", flush=True)

    # ---- 1. forward -----------------------------------------------------
    stage("building inputs")
    x = torch.randn(1, s, h, D, dtype=dt, device=dev)
    ref = x.clone()
    apply_rotary_emb(ref[..., -rd:], freqs)
    stage("reference forward done")
    got = x.clone()
    partial_rope(got, freqs.real, freqs.imag)
    stage("fused forward done")
    lead = (ref[..., :-rd] - got[..., :-rd]).abs().max().item()
    scale = ref[..., -rd:].abs().max().item() or 1.0
    tail = (ref[..., -rd:] - got[..., -rd:]).abs().max().item() / scale
    p1 = lead == 0.0 and tail < 0.02
    ok &= p1
    print(f"\n[{'PASS' if p1 else 'FAIL'}] forward  unrotated maxdiff={lead:.6f} (need 0)  tail rel={tail:.5f} (<0.02)")

    # ---- 2. backward ----------------------------------------------------
    # The reference is called on the slice, so autograd never routes gradient to
    # the unrotated channels; the fused op takes the whole tensor and passes them
    # through.  Compare the rotated tail and check the leading channels are 1.
    base = x.clone().requires_grad_(True)
    y = apply_rotary_emb((base * 1.0)[..., -rd:], freqs)
    stage("reference graph built")
    y.sum().backward()
    stage("reference backward done")
    g_ref = base.grad[..., -rd:]

    g = torch.ones_like(x)
    from cann_ops_transformer import ops as _ops

    r1, r2 = _pack_partial_coefficients(freqs.real, freqs.imag, inverse=False, dtype=dt)
    stage(f"cos/sin cached shapes r1={tuple(r1.shape)} r2={tuple(r2.shape)} dtype={r1.dtype} dev={r1.device}")
    g_out = g.clone()
    _ops.inplace_partial_rotary_mul_backward(g_out, r1, r2, rotary_mode="interleave", partial_slice=[D - rd, D])
    stage("fused backward done")
    rel = (g_ref - g_out[..., -rd:]).abs().max().item() / (g_ref.abs().max().item() or 1.0)
    lead_b = (g_out[..., :-rd].float() - 1.0).abs().max().item()
    p2 = rel < 0.02 and lead_b == 0.0
    ok &= p2
    print(
        f"[{'PASS' if p2 else 'FAIL'}] backward tail rel={rel:.5f} (<0.02)  "
        f"unrotated grad=1.0 maxdiff={lead_b:.6f} (need 0)"
    )

    # ---- 3. inverse -----------------------------------------------------
    xi = torch.randn(1, s, h, D, dtype=dt, device=dev)
    refi = xi.clone()
    apply_rotary_emb(refi[..., -rd:], freqs, True)
    goti = xi.clone()
    partial_rope(goti, freqs.real, freqs.imag, inverse=True)
    lead_i = (refi[..., :-rd] - goti[..., :-rd]).abs().max().item()
    tail_i = (refi[..., -rd:] - goti[..., -rd:]).abs().max().item() / scale
    p3 = lead_i == 0.0 and tail_i < 0.02
    ok &= p3
    print(f"[{'PASS' if p3 else 'FAIL'}] inverse  unrotated maxdiff={lead_i:.6f}  tail rel={tail_i:.5f}")

    # ---- 4. cache must not serve stale values ---------------------------
    # Two different freqs_cis tensors (different theta) must each rotate with
    # their own cos/sin; a stale data_ptr hit would show ~1.0 here.
    f2 = build_freqs(s, rd, dev, theta=40000.0)
    for label, fc in (("theta=1e4", freqs), ("theta=4e4", f2), ("theta=1e4 again", freqs)):
        xa = torch.randn(1, s, h, D, dtype=dt, device=dev)
        refa = xa.clone()
        apply_rotary_emb(refa[..., -rd:], fc)
        partial_rope(xa, fc.real, fc.imag)
        rel4 = (refa[..., -rd:] - xa[..., -rd:]).abs().max().item() / scale
        good = rel4 < 0.02
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] cache    {label:16s} tail rel={rel4:.5f}")

    # ---- 4b. reference must still match production ----------------------
    # Without this, a drift between the reference and what ships would make
    # every numeric result above measure the wrong implementation.
    drift, head = check_production_drift(x, freqs, rd)
    p6 = drift < 0.02 and head == 0.0
    ok &= p6
    print(
        f"[{'PASS' if p6 else 'FAIL'}] prod ref drift={drift:.6f} (<0.02)  head untouched maxdiff={head:.6f} (need 0)"
    )

    # ---- 4c. the cos/sin cache must key on x's dtype ---------------------
    # freqs_cis.dtype is always complex64, so it cannot tell bf16 from fp32;
    # r1/r2 follow the caller's dtype and the operator faults on a mismatch.
    r_bf16, _ = _pack_partial_coefficients(freqs.real, freqs.imag, inverse=False, dtype=torch.bfloat16)
    r_fp32, _ = _pack_partial_coefficients(freqs.real, freqs.imag, inverse=False, dtype=torch.float32)
    p7 = r_bf16.dtype == torch.bfloat16 and r_fp32.dtype == torch.float32
    ok &= p7
    print(f"[{'PASS' if p7 else 'FAIL'}] dtype key bf16->{r_bf16.dtype} fp32->{r_fp32.dtype}")

    # ---- 5. validity guards --------------------------------------------
    p5 = True
    try:
        partial_rope(x, freqs[: s // 2].real, freqs[: s // 2].imag)
        p5 = False
        print("[FAIL] mismatched freqs length was accepted")
    except ValueError:
        print("[PASS] mismatched freqs length rejected")
    # CPU is a supported device now: the wrapper dispatches to the reference
    # implementation instead of rejecting the tensor.
    host = x.cpu()
    rotated_host = partial_rope(host, freqs.cpu().real, freqs.cpu().imag)
    cpu_ok = rotated_host is host and rotated_host[..., :-rd].equal(host[..., :-rd])
    p5 &= cpu_ok
    print(f"[{'PASS' if cpu_ok else 'FAIL'}] CPU tensors use the device fallback")
    ok &= p5

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
