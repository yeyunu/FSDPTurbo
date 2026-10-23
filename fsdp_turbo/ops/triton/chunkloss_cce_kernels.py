# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Triton kernels for CCE vocab-tile streaming loss.

This module owns everything that depends on Triton: the two tile kernels and the
fused autograd Function. It is imported lazily by ops/npu/cce_loss.py so that
the rest of the codebase (including loss_func.py) can run in environments without
Triton installed as long as CCE loss is not enabled.

Ported from a colleague's cce_npu_ms.py + cce_npu_fused.py.
Algorithm origin: Apple Cut-Cross-Entropy (ml-cross-entropy), adapted for NPU.

Key ideas:
  1. Forward: stream logits tile-by-tile along the vocab dimension, merge logsumexp online,
     extract the correct-class logit directly -- never materialize the full (N, V) logits,
     save lse (N,) alongside inputs for backward (CCE's main memory advantage).
  2. Backward: recompute logits tiles from saved lse, compute p = softmax - onehot,
     accumulate dE/dC at tile granularity -- again, no full p materialization.
  3. Multi-stream: cube stream keeps matmul saturated, vector stream runs tile kernels,
     BUFFER_COUNT=3 tile buffers rotate, events enforce ordering.

Replaces the three-layer stack (ChunkLoss + calculate_lm_loss + fixed_cross_entropy) in chunkloss.py.
"""

import torch
import triton
import triton.language as tl


# Number of rotating tile buffer slots. K=2~3 is enough.
# Slot cost: CCE slices along vocab dimension -> slot size = (N, vt),
# decoupled from large V. Three slots cost 3 * N * vt * element_size bytes.
BUFFER_COUNT = 3

# Stream creation on Ascend is expensive (can introduce hundreds of ms bubbles per call).
# Cache streams globally; events are lightweight and created on demand.
_STREAMS = {}


def _get_streams(device):
    key = (device.type, device.index)
    if key not in _STREAMS:
        _STREAMS[key] = (torch.accelerator.Stream(device=device), torch.accelerator.Stream(device=device))
    return _STREAMS[key]


def _tile_ranges(V, vt):
    return [(v0, min(v0 + vt, V)) for v0 in range(0, V, vt)]


@triton.jit
def _fwd_tile_kernel(
    TILE,
    Y,
    M,
    S,
    COR,
    N,
    vt,
    V0,
    stride_tn,
    IGNORE_INDEX: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    """Forward tile kernel: single-pass online softmax + correct-class logit extraction.

    Each program handles a (BR, BV) 2D block. The one-row-per-program version
    achieves only ~100 GB/s (scalar overhead + small vectors); the 2D version
    measures ~370 GB/s.
    """
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < N
    m_in = tl.load(M + rows, mask=rmask, other=-float('inf'))
    s_in = tl.load(S + rows, mask=rmask, other=0.0)
    y = tl.load(Y + rows, mask=rmask, other=IGNORE_INDEX).to(tl.int32)

    mt = tl.full((BR,), -float('inf'), tl.float32)
    st = tl.zeros((BR,), tl.float32)
    cor = tl.zeros((BR,), tl.float32)
    y_loc = y - V0
    # Single-pass online softmax + correct-class logit extraction
    for v0 in range(0, vt, BV):
        offs = v0 + tl.arange(0, BV)
        x = tl.load(
            TILE + rows[:, None] * stride_tn + offs[None, :],
            mask=rmask[:, None] & (offs < vt)[None, :],
            other=-float('inf'),
        ).to(tl.float32)
        nm = tl.maximum(mt, tl.max(x, axis=1))
        st = st * tl.exp(mt - nm) + tl.sum(tl.exp(x - nm[:, None]), axis=1)
        mt = nm
        cor += tl.sum(tl.where(offs[None, :] == y_loc[:, None], x, 0.0), axis=1)

    # Merge (m, s) across tiles
    new_m = tl.maximum(m_in, mt)
    s_new = s_in * tl.exp(m_in - new_m) + st * tl.exp(mt - new_m)
    tl.store(M + rows, new_m, mask=rmask)
    tl.store(S + rows, s_new, mask=rmask)
    valid = (y != IGNORE_INDEX) & (y_loc >= 0) & (y_loc < vt)
    tl.store(COR + rows, tl.load(COR + rows, mask=rmask, other=0.0) + tl.where(valid, cor, 0.0), mask=rmask)


@triton.jit
def _bwd_tile_kernel(
    TILE,
    Y,
    LSE,
    N,
    vt,
    V0,
    stride_tn,
    IGNORE_INDEX: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    """Backward tile kernel: single-pass p=softmax-onehot, write back in-place (bf16).

    Note: on triton-ascend, (vector bool) & (scalar bool) triggers a vector core
    exception. Use 2D where or split where expressions (the valid[:, None] pattern below).
    In-place kernels cannot use triton.autotune -- autotune runs the kernel multiple times,
    and subsequent reads will see already-overwritten gradients.
    """
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < N
    y = tl.load(Y + rows, mask=rmask, other=IGNORE_INDEX).to(tl.int32)
    lse = tl.load(LSE + rows, mask=rmask, other=0.0)
    y_loc = y - V0
    valid = (y != IGNORE_INDEX) & rmask

    for v0 in range(0, vt, BV):
        offs = v0 + tl.arange(0, BV)
        lm = rmask[:, None] & (offs < vt)[None, :]
        x = tl.load(TILE + rows[:, None] * stride_tn + offs[None, :], mask=lm, other=-float('inf')).to(tl.float32)
        p = tl.exp(x - lse[:, None])
        p = tl.where(offs[None, :] == y_loc[:, None], p - 1.0, p)
        p = tl.where(valid[:, None], p, 0.0)
        tl.store(TILE + rows[:, None] * stride_tn + offs[None, :], p.to(TILE.dtype.element_ty), mask=lm)


class ChunkLossCceFused(torch.autograd.Function):
    """CCE vocab-tile streaming + 2D tile fused kernel + dual-stream pipelining.

    Replaces ChunkLoss + calculate_lm_loss + fixed_cross_entropy in chunkloss.py.
    Corresponds to a colleague's cce_npu_ms implementation.

    Args:
        hidden_states: (N, H) bf16, already flattened
        head_weight: (V, H) bf16
        shift_labels: (N,) int64, already flattened
        vt: vocab tile size, recommended 4096
        ignore_index: default -100

    Multi-stream pipeline:
      - matmul_stream: all matmuls (fwd projection / bwd recompute / dC / dE-addmm), cube never idle
      - vector_stream: sequential tile kernels (online softmax state is ordered; grad_hidden addmm is
        naturally ordered within the cube stream)
      - BUFFER_COUNT=3 tile buffers rotate, events enforce ordering:
        wait kernel(i-BUFFER_COUNT) before writing slot, wait matmul(i) before kernel(i),
        wait kernel(i) before dC/dE(i)
      - Prefetch: each iteration enqueues the next tile's matmul before the current tile's gradient matmul
    """

    @staticmethod
    def forward(ctx, hidden_states, head_weight, shift_labels, vt, ignore_index=-100):
        hidden = hidden_states.contiguous()
        weight = head_weight.contiguous()
        labels = shift_labels.contiguous()
        N, D = hidden.shape
        V = weight.shape[0]
        device = hidden.device

        m = torch.full((N,), -float('inf'), dtype=torch.float32, device=device)
        s = torch.zeros(N, dtype=torch.float32, device=device)
        correct = torch.zeros(N, dtype=torch.float32, device=device)

        tiles = _tile_ranges(V, vt)
        T = len(tiles)
        current_stream = torch.accelerator.current_stream(device)
        matmul_stream, vector_stream = _get_streams(device)
        start_event = torch.accelerator.Event()
        start_event.record(current_stream)

        slots = [torch.empty(N, vt, dtype=hidden.dtype, device=device) for _ in range(BUFFER_COUNT)]
        matmul_events = [torch.accelerator.Event() for _ in range(T)]
        kernel_events = [torch.accelerator.Event() for _ in range(T)]

        # Critical: the two loops must be interleaved on the host side.
        # wait_event before record is a no-op. If the cube loop finishes first,
        # then the vec loop runs, all slot-reuse waits become no-ops, and matmul
        # will overwrite slots still being read by the kernel (loss wrong, grad correct).
        for i, (v0, v1) in enumerate(tiles):
            # Cube stream: matmul, wait for kernel(i-BUFFER_COUNT) before reusing the slot
            with torch.accelerator.stream(matmul_stream):
                if i == 0:
                    matmul_stream.wait_event(start_event)
                if i >= BUFFER_COUNT:
                    matmul_stream.wait_event(kernel_events[i - BUFFER_COUNT])
                if v1 - v0 == vt:
                    torch.mm(hidden, weight[v0:v1].t(), out=slots[i % BUFFER_COUNT])
                else:
                    slots[i % BUFFER_COUNT] = hidden @ weight[v0:v1].t()
                matmul_events[i].record(matmul_stream)
            # Vector stream: ordered tile kernel execution (online softmax state is sequential)
            with torch.accelerator.stream(vector_stream):
                if i == 0:
                    vector_stream.wait_event(start_event)
                vector_stream.wait_event(matmul_events[i])
                tile = slots[i % BUFFER_COUNT]
                _fwd_tile_kernel[(triton.cdiv(N, 32),)](
                    tile,
                    labels,
                    m,
                    s,
                    correct,
                    N,
                    v1 - v0,
                    v0,
                    tile.stride(0),
                    IGNORE_INDEX=ignore_index,
                    BR=32,
                    BV=256,
                    num_warps=8,
                )
                kernel_events[i].record(vector_stream)

        current_stream.wait_event(kernel_events[-1])
        lse = m + torch.log(s)
        valid = labels != ignore_index
        loss = torch.where(valid, lse - correct, torch.zeros_like(lse)).sum()

        ctx.save_for_backward(hidden, weight, labels, lse)
        ctx.vt = vt
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, labels, lse = ctx.saved_tensors
        N, D = hidden.shape
        V = weight.shape[0]
        vt = ctx.vt
        device = hidden.device

        tiles = _tile_ranges(V, vt)
        T = len(tiles)
        current_stream = torch.accelerator.current_stream(device)
        matmul_stream, vector_stream = _get_streams(device)
        start_event = torch.accelerator.Event()
        start_event.record(current_stream)

        grad_hidden = torch.zeros(N, D, dtype=torch.float32, device=device)
        grad_weight = torch.empty(V, D, dtype=weight.dtype, device=device)
        slots = [torch.empty(N, vt, dtype=hidden.dtype, device=device) for _ in range(BUFFER_COUNT)]
        matmul_events = [torch.accelerator.Event() for _ in range(T)]
        kernel_events = [torch.accelerator.Event() for _ in range(T)]
        gradient_events = [torch.accelerator.Event() for _ in range(T)]

        # Prefetch tile 0's recompute matmul
        with torch.accelerator.stream(matmul_stream):
            matmul_stream.wait_event(start_event)
            v0, v1 = tiles[0]
            if v1 - v0 == vt:
                torch.mm(hidden, weight[v0:v1].t(), out=slots[0])
            else:
                slots[0] = hidden @ weight[v0:v1].t()
            matmul_events[0].record(matmul_stream)

        with torch.accelerator.stream(vector_stream):
            vector_stream.wait_event(start_event)

        for i, (v0, v1) in enumerate(tiles):
            # Prefetch tile i+1's recompute matmul (wait for the slot's kernel(i+1-BUFFER_COUNT))
            if i + 1 < T:
                nv0, nv1 = tiles[i + 1]
                with torch.accelerator.stream(matmul_stream):
                    if i + 1 >= BUFFER_COUNT:
                        # The slot's previous user is tile(i+1-BUFFER_COUNT)'s dC/dE matmul.
                        # Must wait on gradient_events (after kernel), not kernel_events.
                        matmul_stream.wait_event(gradient_events[i + 1 - BUFFER_COUNT])
                    slot = slots[(i + 1) % BUFFER_COUNT]
                    if nv1 - nv0 == vt:
                        torch.mm(hidden, weight[nv0:nv1].t(), out=slot)
                    else:
                        # Tail tile: non-contiguous out, allocate new tensor
                        slots[(i + 1) % BUFFER_COUNT] = hidden @ weight[nv0:nv1].t()
                    matmul_events[i + 1].record(matmul_stream)

            # Vector stream: tile kernel writes p in-place into the slot
            with torch.accelerator.stream(vector_stream):
                vector_stream.wait_event(matmul_events[i])
                tile = slots[i % BUFFER_COUNT]
                _bwd_tile_kernel[(triton.cdiv(N, 32),)](
                    tile,
                    labels,
                    lse,
                    N,
                    v1 - v0,
                    v0,
                    tile.stride(0),
                    IGNORE_INDEX=ctx.ignore_index,
                    BR=32,
                    BV=256,
                    num_warps=8,
                )
                kernel_events[i].record(vector_stream)

            # Cube stream: wait kernel(i) -> dC matmul -> dE addmm (grad_hidden ordered within the same stream)
            with torch.accelerator.stream(matmul_stream):
                matmul_stream.wait_event(kernel_events[i])
                tile = slots[i % BUFFER_COUNT][:, : v1 - v0] if v1 - v0 < vt else slots[i % BUFFER_COUNT]
                # grad_weight[v0:v1] is written exactly once per backward (each tile covers a
                # distinct vocab range), so weight-dtype storage is safe and avoids the
                # fp32 intermediate + trailing dtype cast.
                torch.mm(tile.t(), hidden, out=grad_weight[v0:v1])
                torch.addmm(grad_hidden, tile, weight[v0:v1], out=grad_hidden)
                gradient_events[i].record(matmul_stream)

        current_stream.wait_event(gradient_events[-1])
        grad_hidden = grad_hidden.to(hidden.dtype)
        grad_hidden = grad_hidden * grad_output
        grad_weight = grad_weight * grad_output
        return grad_hidden, grad_weight, None, None, None
