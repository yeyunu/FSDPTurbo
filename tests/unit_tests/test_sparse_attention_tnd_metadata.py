# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import torch

from fsdp_turbo.ops.npu import sparse_attention


def _call_forward_metadata(cu_seqlens, cu_seqlens_cmp, capture):
    fake_ops = SimpleNamespace(sparse_flash_mla_metadata=lambda **kwargs: capture.update(kwargs) or kwargs)
    with mock.patch.object(sparse_attention, "_get_cann_ops", return_value=fake_ops):
        sparse_attention._forward_metadata(
            batch_size=cu_seqlens.numel() - 1,
            num_query_heads=64,
            num_kv_heads=1,
            head_dim=512,
            topk=512,
            has_compressed_kv=True,
            compress_ratio=2,
            query_length=420,
            original_kv_length=420,
            compressed_kv_length=210,
            residual=torch.tensor([0, 0]),
            window_size=128,
            device_type="cpu",
            device_index=None,
            layout="TND",
            cu_seqlens=cu_seqlens,
            cu_seqlens_cmp=cu_seqlens_cmp,
        )


def test_metadata_normalizes_int64_cumulative_lengths_without_changing_values():
    capture = {}
    cu_seqlens = torch.tensor([0, 410, 830], dtype=torch.int64)
    cu_seqlens_cmp = torch.tensor([0, 205, 415], dtype=torch.int64)

    _call_forward_metadata(cu_seqlens, cu_seqlens_cmp, capture)

    for name, expected in (
        ("cu_seqlens_q", [0, 410, 830]),
        ("cu_seqlens_ori_kv", [0, 410, 830]),
        ("cu_seqlens_cmp_kv", [0, 205, 415]),
    ):
        assert capture[name].dtype == torch.int32
        assert capture[name].device == cu_seqlens.device
        assert capture[name].tolist() == expected


def test_metadata_keeps_existing_int32_cumulative_lengths():
    capture = {}
    cu_seqlens = torch.tensor([0, 296, 632, 998], dtype=torch.int32)
    cu_seqlens_cmp = torch.tensor([0, 148, 316, 499], dtype=torch.int32)

    _call_forward_metadata(cu_seqlens, cu_seqlens_cmp, capture)

    assert capture["cu_seqlens_q"] is cu_seqlens
    assert capture["cu_seqlens_ori_kv"] is cu_seqlens
    assert capture["cu_seqlens_cmp_kv"] is cu_seqlens_cmp


def test_backward_metadata_also_normalizes_cumulative_lengths():
    capture = {}
    fake_ops = SimpleNamespace(sparse_flash_mla_grad_metadata=lambda **kwargs: capture.update(kwargs) or kwargs)
    with mock.patch.object(sparse_attention, "_get_cann_ops", return_value=fake_ops):
        sparse_attention._backward_metadata(
            batch_size=2,
            query_length=420,
            original_kv_length=420,
            compressed_kv_length=210,
            num_query_heads=64,
            num_kv_heads=1,
            head_dim=512,
            topk=512,
            has_compressed_kv=True,
            compress_ratio=2,
            residual=torch.tensor([0, 0]),
            window_size=128,
            device_type="cpu",
            device_index=None,
            layout="TND",
            cu_seqlens=torch.tensor([0, 410, 830], dtype=torch.int64),
            cu_seqlens_cmp=torch.tensor([0, 205, 415], dtype=torch.int64),
        )

    assert capture["cu_seqlens_q"].dtype == torch.int32
    assert capture["cu_seqlens_ori_kv"].dtype == torch.int32
    assert capture["cu_seqlens_cmp_kv"].dtype == torch.int32
    assert capture["cu_seqlens_cmp_kv"].tolist() == [0, 205, 415]
