# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import types
import warnings
from typing import Callable, Any
from collections import defaultdict

import torch

from fsdp_turbo.ops.quantize import w4a16_quant_matmul
from fsdp_turbo.quantization.mxfp4_config import float4_dtype


_weight_cache: dict[int, Any] = {}
_num_quantized: int = 0
_num_cache_hits: int = 0
_opt_hooked: bool = False
_identity_linear_cache = {}
_identity_gmm_cache = {}
_cached_weight_pool = defaultdict(lambda: {})


def cached_weight_fakequant(
    x: torch.Tensor,
    quantizer: Callable[..., Any],
    bias: torch.Tensor = None,
    block_size: int = 32,
    key: torch.Tensor = None,
    **kwargs,
):
    global _num_quantized, _num_cache_hits

    key = x if key is None else key

    if key in _cached_weight_pool:
        _num_cache_hits += 1
        return _cached_weight_pool[key]

    if len(x.shape) == 3:
        e, m, n = x.shape
        if m not in _identity_gmm_cache:
            identity_matrix = torch.eye(m, dtype=torch.bfloat16, device=x.device)
            identity_quant, identity_sf_quant = quantizer(identity_matrix, axis=-1, **kwargs)
            _identity_gmm_cache[m] = (identity_quant, identity_sf_quant)

        identity_quant, identity_sf_quant = _identity_gmm_cache[m]

        weight_2d = x.permute(1, 0, 2).reshape(m, e * n)
        weight_quant, weight_sf_quant = quantizer(weight_2d, axis=-2, **kwargs)

        output = w4a16_quant_matmul(
            identity_quant,
            weight_quant,
            weight_sf_quant,
            pertoken_scale=identity_sf_quant,
            bias=bias,
            scale_dtype=torch.float8_e8m0fnu,
            pertoken_scale_dtype=torch.float8_e8m0fnu,
            output_dtype=torch.bfloat16,
            x1_dtype=float4_dtype(),
            x2_dtype=float4_dtype(),
            group_sizes=[1, 1, 32],
        )

        weight = output.reshape(m, e, n).permute(1, 0, 2)
    else:
        n = x.shape[-1]
        if n not in _identity_linear_cache:
            identity_matrix = torch.eye(n, dtype=torch.bfloat16, device=x.device)
            identity_quant, identity_sf_quant = quantizer(identity_matrix, axis=-2, **kwargs)
            _identity_linear_cache[n] = (identity_quant, identity_sf_quant)

        identity_quant, identity_sf_quant = _identity_linear_cache[n]
        weight_quant, weight_sf_quant = quantizer(x, axis=-1, **kwargs)

        output = w4a16_quant_matmul(
            weight_quant,
            identity_quant,
            identity_sf_quant,
            pertoken_scale=weight_sf_quant,
            bias=bias,
            scale_dtype=torch.float8_e8m0fnu,
            pertoken_scale_dtype=torch.float8_e8m0fnu,
            output_dtype=torch.bfloat16,
            x1_dtype=float4_dtype(),
            x2_dtype=float4_dtype(),
            group_sizes=[1, 1, 32],
        )

        weight = output

    _cached_weight_pool[key] = weight
    _num_quantized += 1

    return weight


def cached_weight_quant(
    weight: torch.Tensor,
    quantizer: Callable[[torch.Tensor, Any], Any],
    *args,
    cache_source: torch.Tensor | None = None,
    **kwargs,
) -> Any:
    global _num_quantized, _num_cache_hits

    if not _opt_hooked:
        warnings.warn(
            "optimizer.step is not hooked, quantizer will be called without caching, "
            "which will impair the training performance"
        )
        return quantizer(weight, *args, **kwargs)

    source = weight if cache_source is None else cache_source
    key = (
        source.device,
        source.untyped_storage().data_ptr(),
        source.storage_offset(),
        tuple(source.shape),
        id(quantizer),
    )

    if key in _weight_cache:
        _num_cache_hits += 1
        return _weight_cache[key]

    result = quantizer(weight, *args, **kwargs)
    _weight_cache[key] = result
    _num_quantized += 1
    return result


def clear_weight_cache() -> None:
    global _num_quantized, _num_cache_hits
    for val in _weight_cache.values():
        if isinstance(val, torch.Tensor):
            val.untyped_storage().resize_(0)
        elif isinstance(val, tuple):
            for t in val:
                if isinstance(t, torch.Tensor):
                    t.untyped_storage().resize_(0)
    _weight_cache.clear()
    _cached_weight_pool.clear()
    _identity_linear_cache.clear()
    _identity_gmm_cache.clear()
    _num_quantized = 0
    _num_cache_hits = 0


def hook_optimizer_step(model, optimizer) -> None:
    global _opt_hooked
    if _opt_hooked:
        return

    original_step = optimizer.step

    def step_with_cache_clear(self, *args, **kwargs):
        clear_weight_cache()
        return original_step(*args, **kwargs)

    optimizer.step = types.MethodType(step_with_cache_clear, optimizer)
    _opt_hooked = True
