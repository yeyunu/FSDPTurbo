# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU contract checks for the public AscendC KDA adapter, without NPU kernels."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("disable_recompute", [False, True])
@pytest.mark.parametrize("sigmoid", [False, True])
def test_kda_public_layout_and_beta_gradient(
    platform: str,
    packed: bool,
    disable_recompute: bool,
    sigmoid: bool,
) -> None:
    """Keep scalar/gate tensors token-major and apply sigmoid backward once."""
    calls = []

    def ascendc_forward(query, key, value, gate, beta, scale, chunk_size, **kwargs):
        assert kwargs["layout"] == "BSND"
        assert not kwargs.get("use_beta_sigmoid_in_kernel", False)
        assert not kwargs.get("allow_neg_eigval", False)
        assert kwargs.get("beta_eff_out") is None
        for name, source in (("q_hat_out", query), ("k_hat_out", key)):
            if kwargs.get(name) is not None:
                kwargs[name].copy_(source.transpose(1, 2))
        for name in ("q_rstd_out", "k_rstd_out"):
            if kwargs.get(name) is not None:
                kwargs[name].fill_(1)
        head = query.transpose(1, 2).contiguous()
        output = query + key + value + beta[..., None] / query.shape[-1]
        caches = [head.float(), head, head, head, head, head, head, head, head, None]
        return (output.to(query.dtype), None, *caches)

    def ascendc_backward(
        query,
        key,
        value,
        beta,
        gate,
        aqk,
        akk,
        weight,
        query_gate,
        key_gate,
        new_value,
        state,
        output_gradient,
        scale,
        *,
        implementation="auto",
        **kwargs,
    ):
        calls.append(implementation)
        batch, tokens, heads, dim = shape
        expected_gate = (tokens, heads, dim) if packed else shape
        expected_beta = expected_gate[:-1]
        expected_token = (heads, tokens, dim) if packed else (batch, heads, tokens, dim)
        assert query.shape == key.shape == value.shape == expected_token
        assert output_gradient.shape == kwargs["raw_g"].shape == expected_gate
        assert beta.shape == expected_beta
        assert kwargs["layout"] == ("NTD" if packed else "BNSD")
        assert implementation == ("optimized" if platform == "a5" else "legacy")
        assert kwargs["disable_recompute"] == (disable_recompute if platform == "a5" else True)
        assert not kwargs.get("use_beta_sigmoid_in_kernel", False)
        assert kwargs.get("beta_eff") is None
        if platform == "a5":
            assert kwargs["q_rstd"] is not None and kwargs["k_rstd"] is not None
        expected = effective_beta.squeeze(0) if packed else effective_beta
        torch.testing.assert_close(beta, expected)
        bias = kwargs["dt_bias"]
        return (
            torch.ones_like(query),
            torch.ones_like(key),
            torch.ones_like(value),
            torch.ones_like(beta),
            torch.ones_like(kwargs["raw_g"]),
            None,
            torch.zeros_like(kwargs["A_log"]),
            torch.zeros_like(bias),
        )

    ascendc = types.ModuleType("fla_npu.ops.ascendc")
    ascendc.chunk_kda_fwd = ascendc_forward
    ascendc.chunk_kda_bwd = ascendc_backward
    triton = types.ModuleType("triton_ascend_kernels.attention.fla.kda.chunk")
    triton.l2norm_fwd = lambda value: (value, torch.ones(value.shape[:-1]))
    triton.l2norm_bwd = lambda value, inverse_std, gradient: gradient
    triton.chunk_kda_fwd = mock.Mock(side_effect=AssertionError("Unexpected Triton forward"))
    triton.chunk_kda_bwd = mock.Mock(side_effect=AssertionError("Unexpected Triton backward"))
    triton.prepare_chunk_indices = mock.Mock(side_effect=AssertionError("Unexpected chunk indices"))
    for name in ("input_guard", "autocast_custom_fwd", "autocast_custom_bwd"):
        setattr(triton, name, lambda function: function)
    path = Path(__file__).resolve().parents[3] / "fsdp_turbo/ops/npu/chunk_kda_ascendc.py"
    spec = importlib.util.spec_from_file_location("_kda_contract_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as modules:
        modules.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)
        modules.setitem(sys.modules, "triton_ascend_kernels.attention.fla.kda.chunk", triton)
        spec.loader.exec_module(adapter)

    shape = (1, 4, 8, 128)
    query, key, value = [torch.randn(shape, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    gate = torch.randn(shape, requires_grad=True)
    beta = torch.randn(shape[:-1], requires_grad=True)
    a_log = torch.zeros(shape[2], requires_grad=True)
    bias = torch.zeros(shape[2] * shape[3], requires_grad=True)
    effective_beta = 2 * beta.sigmoid() if sigmoid else beta
    caches = tuple(torch.zeros((1, 8, 4, 128)) for _ in range(6))
    with (
        mock.patch.object(adapter, "_ascend_platform", return_value=platform),
        mock.patch.object(
            adapter,
            "_a2a3_recompute",
            return_value=caches,
        ) as recompute,
    ):
        output, _ = adapter.chunk_kda_ascendc(
            query,
            key,
            value,
            gate,
            beta,
            A_log=a_log,
            dt_bias=bias,
            safe_gate=True,
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=sigmoid,
            allow_neg_eigval=sigmoid,
            disable_recompute=disable_recompute,
            cu_seqlens=torch.tensor([0, shape[1]]) if packed else None,
        )
        output.float().sum().backward()
    assert calls == ["optimized" if platform == "a5" else "legacy"]
    assert recompute.call_count == int(platform == "a2a3" and not disable_recompute)
    expected = 2 * beta.sigmoid() * (1 - beta.sigmoid()) if sigmoid else torch.ones_like(beta)
    torch.testing.assert_close(beta.grad, expected)
    assert query.grad.shape == key.grad.shape == value.grad.shape == gate.grad.shape == shape
