# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Flash-based causal attention for Kimi K3 MLA under context parallelism.

The eager einsum path materializes ``[B, H, S, S]`` score/prob tensors plus a
``[B, 1, S, S]`` additive causal mask, which is infeasible beyond ~100k tokens
(2 TiB of mask alone at 1M). This module expresses causality with the flash
kernel's native causal mode instead:

* NPU: ``torch_npu.npu_fusion_attention`` with ``sparse_mode=3``. The
  ``atten_mask`` argument is a cached 2048x2048 bool placeholder whose content
  the kernel ignores under ``sparse_mode=3``, so mask cost never grows with S.
* CUDA/CPU: ``torch.nn.functional.scaled_dot_product_attention(is_causal=True)``.

The backend split lives in ``fsdp_turbo.ops.flash_attn.causal_flash_attention``
behind the operator registry (dev-guide rule 3: the model side only calls the
semantic op), and the SDPA reference keeps bottom-right causal alignment for a
query block against a longer K/V prefix.

Under Ulysses/KCP this function runs after the all-to-all on the complete
sequence, so a plain causal flag is semantically identical to the dense causal
mask the eager wrapper used to rebuild. Right padding is assumed
(the CP data path pads the sequence tail): real tokens never attend to later
padded tokens under a causal mask, and padded rows are dropped from the loss
via ``ignore_index`` and zeroed by the KDA output mask.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fsdp_turbo.ops.flash_attn import causal_flash_attention


CP_EXCHANGED_KWARG = "_fsdp_turbo_cp_exchanged"
FLASH_CAUSAL_MARKER = "_fsdp_turbo_flash_causal"

# The model-side attention dispatcher; the CP wrappers are installed over this
# single surface (see cp_plan.<mode>_function_patches).
ATTENTION_PATCH_TARGET = "modeling_kimi_linear.dispatch_attention"


def _active_context_parallel_size():
    """Best-effort read of the active CP size; 1 when CP is off or unknown."""
    if not dist.is_initialized():
        return 1
    try:
        from fsdp_turbo.distributed.parallel_state import ParallelState, get_parallel_state

        if not ParallelState._instances:
            return 1
        parallel_state = get_parallel_state()
        kcp_size = getattr(parallel_state, "kcp_parallel_size", 1) or 1
        try:
            ulysses_size = parallel_state.get_ulysses_group_size()
        except Exception:
            ulysses_size = 1
        try:
            kv_allgather_size = parallel_state.get_kvallgather_group_size()
        except Exception:
            kv_allgather_size = 1
        return max(kcp_size, ulysses_size or 1, kv_allgather_size or 1)
    except Exception:
        return 1


def context_parallel_active():
    """Whether this process belongs to a sequence-sharded context-parallel run.

    ``KimiMLAAttention.forward`` branches on this: under sequence parallelism
    the attention path dispatches to the flash kernel (the dense mask and eager
    score matrix are infeasible once the sequence is split for long contexts);
    single-rank runs keep the upstream config-driven implementation.
    """
    return _active_context_parallel_size() > 1


def _cp_exchange_guard(cp_exchanged: bool):
    """Fail fast when sequence-sharded inputs bypassed the CP exchange."""
    if cp_exchanged or _active_context_parallel_size() <= 1:
        return
    raise ValueError(
        "Kimi flash attention received sequence-sharded inputs under context "
        "parallelism without a CP exchange. The flash kernel must run behind the "
        "Ulysses all-to-all wrapper: add "
        f"{ATTENTION_PATCH_TARGET} (type: full_attention) to "
        "cp_plan.ulysses_function_patches / cp_plan.kcp_function_patches in "
        "the training config."
    )


def kimi_flash_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
):
    """Drop-in flash replacement for ``eager_attention_forward`` (Kimi MLA).

    Accepts the unified call surface of ``KimiMLAAttention.forward`` (module,
    query, key, value, attention_mask, dropout, scaling, ...) and returns the
    ``(attn_output, attn_weights)`` pair. Causality comes from the kernel's
    native causal mode; a 2D padding mask is accepted and ignored (right
    padding). A dense 4-D mask is rejected rather than silently dropped.
    """
    cp_exchanged = kwargs.pop(CP_EXCHANGED_KWARG, False)
    _cp_exchange_guard(cp_exchanged)

    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)

    if query.dim() != 4:
        raise ValueError(f"Kimi flash attention expects [batch, heads, seq, head_dim]; got {tuple(query.shape)}.")

    if key.shape[1] != query.shape[1]:
        repeats = query.shape[1] // key.shape[1]
        key = torch.repeat_interleave(key, dim=1, repeats=repeats)
        value = torch.repeat_interleave(value, dim=1, repeats=repeats)

    head_dim = query.shape[-1]
    value_head_dim = value.shape[-1]

    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    if attention_mask is not None and attention_mask.dim() > 2:
        raise ValueError(
            "Kimi flash attention does not accept a dense 4-D mask; causality comes from the "
            "kernel's native causal mode. Run without context parallelism for the eager path."
        )

    if value_head_dim != head_dim:
        if value_head_dim > head_dim:
            raise ValueError(
                "Kimi flash attention expects value head_dim <= query head_dim; "
                f"got value {value_head_dim} > query {head_dim}."
            )
        # The flash kernels require the value head dim to match q/k; the model
        # no longer pads on its side (that path belonged to the legacy FA2
        # branch), so the kernel wrapper pads and slices back afterwards.
        value = F.pad(value, [0, head_dim - value_head_dim])

    # Backend differences (NPU npu_fusion_attention sparse_mode=3 vs SDPA)
    # live in the registered op; causality is native to the kernel, including
    # bottom-right alignment for a query block against a longer K/V prefix.
    attn_output = causal_flash_attention(query, key, value, scaling=scaling, dropout=dropout, is_causal=is_causal)
    if value_head_dim != head_dim:
        attn_output = attn_output[..., :value_head_dim]
    # ``eager_attention_forward`` returns [batch, seq, heads, head_dim]; the
    # Ulysses output all-to-all and KimiMLAAttention's reshape both rely on
    # that layout.
    return attn_output.transpose(1, 2), None


# Read by the Ulysses wrapper to skip dense-mask construction for this kernel.
kimi_flash_attention_forward.__dict__[FLASH_CAUSAL_MARKER] = True
