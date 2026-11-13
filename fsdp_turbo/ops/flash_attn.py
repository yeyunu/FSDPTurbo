# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Flash attention forward with automatic device dispatch.

Public entry point that dispatches to the NPU ``npu_fusion_attention`` or
CPU eager fallback via the operator registry.  The registered function is
also registered as the transformers ``flash_attention_2`` backend so that
models with ``_attn_implementation="flash_attention_2"`` use this path.

Query/key/value arrive in **BNSD** layout ``[B, H, S, D]``.
"""

import logging

import torch

from fsdp_turbo.ops.registry import dispatch_op
from fsdp_turbo.utils.device import get_accelerator_name
from fsdp_turbo.utils.log import log_warning_once

logger = logging.getLogger(__name__)

__all__ = ["causal_flash_attention", "flash_attention_forward", "register_flash_attention"]


def causal_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float | None = None,
    dropout: float = 0.0,
    is_causal: bool = True,
) -> torch.Tensor:
    """Native-causal flash attention for BNSD tensors, dispatched by the query device.

    Causality comes from the kernel itself -- NPU ``npu_fusion_attention`` with
    ``sparse_mode=3``, SDPA ``is_causal`` on CPU/CUDA -- so no dense
    ``[B, 1, S, S]`` mask is ever materialized. When the query block is shorter
    than the K/V prefix, causality stays bottom-right aligned (row ``i``
    attends ``K[0, i + kv_len - q_len)``). Inputs must already have matching
    head counts and head dims. Dispatch uses the input tensor's device, including
    CPU tensors on an accelerator host. Returns the output in BNSD layout.
    """
    return dispatch_op(
        "causal_flash_attention",
        query,
        key,
        value,
        scaling=scaling,
        dropout=dropout,
        is_causal=is_causal,
        device_type=query.device.type,
    )


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    scaling: float | None = None,
    dropout: float = 0.0,
    is_causal: bool = True,
    **kwargs: object,
) -> tuple[torch.Tensor, None]:
    """Compute attention via device-dispatched fused op or eager fallback.

    Args:
        module: The attention module (provides ``config``, ``layer_idx``).
        query/key/value: ``[B, H, S, D]`` tensors (BNSD layout).
        attention_mask: 4-D additive-float ``[B, 1, S, S]`` mask, or ``None``.
        scaling: Softmax scale (``head_dim ** -0.5``).
        dropout: Dropout probability.
        is_causal: Whether the original attention was causal.

    Returns:
        Tuple ``(attn_output, None)``.
    """
    return dispatch_op(
        "flash_attention",
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        dropout=dropout,
        is_causal=is_causal,
        device_type=query.device.type,
        **kwargs,
    )


_FLASH_ATTN_REGISTERED = False


def register_flash_attention():
    """Register :func:`flash_attention_forward` as the ``flash_attention_2`` backend."""
    global _FLASH_ATTN_REGISTERED
    if _FLASH_ATTN_REGISTERED or get_accelerator_name() != "npu":
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)

        _FLASH_ATTN_REGISTERED = True
    except (ImportError, AttributeError) as exc:
        log_warning_once(logger, f"flash_attention_2 backend not registered: {exc}")


def vision_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_endpoints: torch.Tensor | tuple[int, ...],
    key_endpoints: torch.Tensor | tuple[int, ...],
) -> torch.Tensor:
    """Bidirectional TND vision attention using cumulative sequence endpoints.

    Endpoints exclude the initial zero on the native NPU path, matching K3's
    established encoder contract. Returns a tensor with the query layout.
    """
    return dispatch_op(
        "vision_attention", query, key, value, query_endpoints, key_endpoints, device_type=query.device.type
    )
