# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU coverage for SLIG routing, scaling, packed boundaries and replay."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

from fsdp_turbo.memory.chunk_batch.chunk_batch import chunk_mbs_forward
from fsdp_turbo.models.deepseek.deepseek_v41.attention import Attention
from fsdp_turbo.models.deepseek.deepseek_v41.model_args import ModelArgs
from fsdp_turbo.models.deepseek.deepseek_v41.runtime import shared_attn
from fsdp_turbo.ops.sparse_indexer_loss import (
    collect_indexer_gradients,
    record_indexer_gradient,
)


@pytest.fixture(autouse=True)
def cpu_semantic_dispatch(monkeypatch):
    """Keep small CPU fixtures independent of the host's default accelerator."""
    from fsdp_turbo.ops import registry

    get_op = registry.get_op
    monkeypatch.setattr(
        registry,
        "get_op",
        lambda name, device_type=None: get_op(name, "cpu" if device_type is None else device_type),
    )


def test_independent_teacher_and_gradient_oracle():
    torch.manual_seed(7)
    batch, length, heads, dim = 2, 9, 3, 4
    q = torch.randn(batch, length, heads, dim)
    kv = torch.randn(batch, length, dim)
    cmp = torch.randn(batch, 4, dim)
    sinks = torch.randn(heads)
    win = torch.arange(length).view(1, length, 1).expand(batch, -1, -1)
    selected = torch.arange(4).expand(batch, length, -1)
    selected = selected.masked_fill(selected >= (win + 1) // 2, -1)
    iq = torch.randn_like(q, requires_grad=True)
    ik = torch.randn_like(cmp, requires_grad=True)
    w = torch.randn(batch, length, heads, requires_grad=True)
    with collect_indexer_gradients() as terms:
        record_indexer_gradient(q, kv, cmp, win, selected, (iq, ik, w), sinks, 0.5, 2, False, None)
    assert terms[0].item() == 0
    actual = torch.autograd.grad(terms[0] / 3, (iq, ik, w))
    oracle = 0
    for b in range(batch):
        for s in range(1, length):
            count = (s + 1) // 2
            keys = torch.cat((kv[b, s : s + 1], cmp[b, :count]))
            attn = torch.cat((q[b, s] @ keys.T * 0.5, sinks[:, None]), 1).softmax(-1)
            teacher = attn[:, 1:-1].sum(0)
            teacher /= teacher.sum()
            logits = ((iq[b, s] @ ik[b, :count].T).relu() * w[b, s, :, None]).sum(0)
            oracle -= (teacher * logits.log_softmax(-1)).sum()
    for a, e in zip(actual, torch.autograd.grad(oracle / 3, (iq, ik, w))):
        assert e.abs().max() > 1e-3
        torch.testing.assert_close(a, e, atol=3e-7, rtol=2e-5)


def _layers():
    args = ModelArgs(
        dim=8,
        head_dim=4,
        rope_head_dim=2,
        q_lora_rank=4,
        n_heads=2,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=3,
        o_groups=1,
        o_lora_rank=4,
        compress_ratios=(2, 2),
        kv_source_layers=(0,),
        index_source_layers=(0,),
        n_layers=2,
        n_mtp_layers=0,
        max_seq_len=32,
        window_size=3,
        indexer_loss_coeff=0.3,
    )
    layers = torch.nn.ModuleList(Attention(i, args) for i in range(2))
    torch.manual_seed(44)
    with torch.no_grad():
        for param in layers.parameters():
            param.normal_(0, 0.3)
    return layers


def _run(layers, x, *, packed=False, chunk=False, replay=False, denom=None):
    shared_attn.reset()
    with collect_indexer_gradients() as terms:
        for layer in layers:
            forward = layer.forward
            if chunk:
                forward = chunk_mbs_forward(1, chunk_arg_indexes=[0])(forward)
            kw = dict(sfa_layout="TND", cu_seqlens=torch.tensor([0, 4, 10])) if packed else {}
            if replay:

                def context():
                    return nullcontext(), shared_attn.replay_window(shared_attn.snapshot())

                x = checkpoint(forward, x, 0, use_reentrant=False, context_fn=context, **kw)
            else:
                x = forward(x, 0, **kw)
    assert len(terms) >= 2
    (sum(terms) * (0.3 / (denom or x.size(0) * x.size(1)))).backward()
    return x.detach()


@pytest.mark.parametrize("chunk,replay", [(True, False), (False, True), (True, True)])
def test_shared_indexer_chunk_and_replay(chunk, replay):
    layers = _layers()
    x = torch.randn(3, 10, 8, dtype=torch.bfloat16)
    baseline = _run(layers, x)
    expected = [p.grad.clone() for p in layers[0].indexer.parameters() if p.grad is not None]
    layers.zero_grad(set_to_none=True)
    actual = _run(layers, x, chunk=chunk, replay=replay)
    torch.testing.assert_close(actual, baseline)
    grads = [p.grad for p in layers[0].indexer.parameters() if p.grad is not None]
    assert len(grads) == len(expected) > 0
    for a, e in zip(grads, expected):
        assert e.abs().max() > 0
        torch.testing.assert_close(a, e, atol=2e-5, rtol=0.05)


def test_packed_matches_accumulated_documents(monkeypatch):
    from fsdp_turbo.models.deepseek.deepseek_v41 import attention
    from fsdp_turbo.ops.cpu.sparse_attention import indexed_sparse_attention_torch

    def sfa_reference(query, kv, sinks, window, scale, **kwargs):
        # Emulate the SFA boundary on CPU to exercise packed SLIG slicing.
        assert kwargs["use_sparse_flash_attn"]
        compressed = kwargs["compressed_key_value"]
        selected = kwargs["compressed_topk_indices"]
        ratio = kwargs["compress_ratio"]
        if kwargs["layout"] == "BSND":
            return indexed_sparse_attention_torch(
                query,
                kv,
                sinks,
                window,
                scale,
                compressed_key_value=compressed,
                compressed_topk_indices=selected,
            )
        bounds = kwargs["cu_seqlens"].tolist()
        outputs = []
        key_start = 0
        for start, end in zip(bounds[:-1], bounds[1:]):
            length = end - start
            count = length // ratio
            local_window = attention.get_window_topk_idxs(kwargs["window_size"], 1, length, 0, query.device)
            outputs.append(
                indexed_sparse_attention_torch(
                    query[None, start:end],
                    kv[None, start:end],
                    sinks,
                    local_window,
                    scale,
                    compressed_key_value=compressed[None, key_start : key_start + count],
                    compressed_topk_indices=selected[None, start:end],
                )
            )
            key_start += count
        return torch.cat(outputs, dim=1).squeeze(0)

    monkeypatch.setattr(attention, "indexed_sparse_attention", sfa_reference)
    layers = _layers()
    for layer in layers:
        layer.use_sparse_flash_attn = True
    x = torch.randn(1, 10, 8, dtype=torch.bfloat16)
    out = _run(layers, x, packed=True)
    expected = layers[0].indexer.wq_b.weight.grad.clone()
    layers.zero_grad(set_to_none=True)
    separate = torch.cat([_run(layers, doc, denom=10) for doc in x.split([4, 6], 1)], 1)
    torch.testing.assert_close(out, separate)
    assert expected.norm() > 0
    torch.testing.assert_close(layers[0].indexer.wq_b.weight.grad, expected, atol=2e-5, rtol=0.05)


def _cp_worker(rank, path):
    from datetime import timedelta
    import torch.distributed as dist
    from fsdp_turbo.distributed.context_parallel.loss import (
        cp_all_reduce_forward_backward,
    )

    dist.init_process_group("gloo", init_method=path, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        layers = _layers()
        x = torch.randn(1, 12, 8, dtype=torch.bfloat16)
        mask = torch.arange(12).reshape(1, -1) < 10
        one = [dist.new_group([member]) for member in range(2)][rank]

        def run(inputs, group, query_mask):
            state = None
            with collect_indexer_gradients() as terms:
                for layer in layers:
                    inputs, state = layer(
                        inputs,
                        0,
                        cp_group=group,
                        cp_state=state,
                        attention_mask=query_mask,
                    )
            (cp_all_reduce_forward_backward(sum(terms), group) / mask.sum()).backward()
            return [p.grad.float().clone() for p in layers[0].indexer.parameters() if p.grad is not None]

        expected = run(x, one, mask)
        layers.zero_grad(set_to_none=True)
        grads = run(x.chunk(2, 1)[rank], dist.group.WORLD, mask.chunk(2, 1)[rank])
        for actual, reference in zip(grads, expected):
            dist.all_reduce(actual)
            actual /= 2
            assert reference.norm() > 0
            torch.testing.assert_close(actual, reference, atol=0.0002, rtol=0.05)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo required")
def test_two_rank_cp_gradient(tmp_path):
    torch.multiprocessing.spawn(_cp_worker, args=((tmp_path / "cp-store").as_uri(),), nprocs=2)


def _adapter_args():
    return ModelArgs(
        dim=8,
        head_dim=4,
        rope_head_dim=2,
        q_lora_rank=4,
        n_heads=2,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=3,
        o_groups=1,
        o_lora_rank=4,
        compress_ratios=(2, 2),
        kv_source_layers=(0,),
        index_source_layers=(0,),
        n_layers=2,
        n_mtp_layers=0,
        max_seq_len=32,
        window_size=3,
        vocab_size=32,
        moe_inter_dim=8,
        n_routed_experts=2,
        n_activated_experts=1,
    )


@pytest.mark.parametrize("sfa", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_adapter_accumulation_preserves_lm_loss_and_indexer_gradient(sfa, fused):
    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import build_deepseek_v41_model

    args = _adapter_args()
    kwargs = dict(model_args=args, use_sparse_flash_attn=sfa, use_fused_lightning_indexer_loss=fused)
    model = build_deepseek_v41_model(**kwargs, indexer_loss_coeff=0.3)
    plain = build_deepseek_v41_model(**kwargs, indexer_loss_coeff=0)
    torch.manual_seed(17)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.15)
    plain.load_state_dict(model.state_dict())
    ids = torch.randint(0, 32, (2, 10))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, -2:] = False

    def run(net, tokens, valid):
        return net(tokens, attention_mask=valid, labels=tokens).loss

    loss = run(model, ids, mask)
    torch.testing.assert_close(loss, run(plain, ids, mask), rtol=0, atol=0)
    loss.backward()
    indexer = model.model.layers[0].attn.indexer
    expected = indexer.wq_b.weight.grad.clone()
    assert expected.norm() > 0
    model.zero_grad(set_to_none=True)
    for row, valid in zip(ids.split(1), mask.split(1)):
        (run(model, row, valid) / 2).backward()
    torch.testing.assert_close(indexer.wq_b.weight.grad, expected, atol=2e-6, rtol=0.06)


def _tp_worker(rank, path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import build_deepseek_v41_model
    from fsdp_turbo.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
    from fsdp_turbo.fsdp_turbo_config import TPPlanConfig

    dist.init_process_group("gloo", init_method=path, rank=rank, world_size=2)
    try:
        model = build_deepseek_v41_model(model_args=_adapter_args(), indexer_loss_coeff=0.3)
        torch.manual_seed(32)
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.2)
        ids = torch.randint(0, 32, (2, 10))
        out = model(ids, labels=ids)
        out.loss.backward()
        baseline = model.model.layers[0].attn.indexer.wq_b.weight.grad.clone()
        assert baseline.norm() > 0
        model.zero_grad(set_to_none=True)
        ffn = "model.layers.{*}.ffn.shared_experts."
        plan = TPPlanConfig([ffn + "w1", ffn + "w3"], [ffn + "w2"], [])
        tensor_parallel_modules(model, init_device_mesh("cpu", (2,)), plan)
        out_tp = model(ids, labels=ids)
        out_tp.loss.backward()
        torch.testing.assert_close(out_tp.loss, out.loss, atol=0.005, rtol=0.005)
        torch.testing.assert_close(model.model.layers[0].attn.indexer.wq_b.weight.grad, baseline, atol=2e-5, rtol=0.05)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo required")
def test_two_rank_tp_gradient(tmp_path):
    torch.multiprocessing.spawn(_tp_worker, args=((tmp_path / "tp-store").as_uri(),), nprocs=2)


def test_real_recompute_and_chunk_wrappers_on_adapter():
    from fsdp_turbo.memory.chunk_batch.chunk_batch import chunk_batch_modules
    from fsdp_turbo.memory.recompute.recompute import recompute_modules
    from fsdp_turbo.fsdp_turbo_config import ChunkBatchPlanConfig
    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import build_deepseek_v41_model

    model = build_deepseek_v41_model(model_args=_adapter_args(), indexer_loss_coeff=0.4)
    torch.manual_seed(32)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.2)
    ids = torch.randint(0, 32, (3, 10))
    mask = torch.arange(10)[None, :] < torch.tensor([[10], [8], [6]])
    plain = model(ids, labels=ids, attention_mask=mask).loss
    plain.backward()
    params = list(model.model.layers[0].attn.indexer.parameters())
    expected = [p.grad.clone() for p in params]
    model.zero_grad(set_to_none=True)
    targets = ["model.layers.{*}"]
    recompute_modules(model, targets)
    chunk_batch_modules(
        model,
        ChunkBatchPlanConfig(
            apply_modules=targets,
            chunk_mbs=2,
            chunk_arg_indexes=[0, 2, 3],
            chunk_kwarg_names=["attention_mask", "engram_mask"],
        ),
    )
    actual = model(ids, labels=ids, attention_mask=mask).loss
    actual.backward()
    torch.testing.assert_close(actual, plain, atol=0.002, rtol=0.002)
    for p, ref in zip(params, expected):
        assert ref.norm() > 0
        assert (p.grad.float() - ref.float()).norm() / ref.float().norm() < 0.05


@pytest.mark.parametrize("use_fused_kl", [False, True])
@pytest.mark.parametrize("layout", ["bsnk", "bsk", "swapped"])
def test_npu_wrapper_normalizes_teacher_and_preserves_zero_rows(monkeypatch, layout, use_fused_kl):
    """Check the native call boundary without needing an Ascend device."""
    import sys
    from types import SimpleNamespace
    from fsdp_turbo.ops.npu import sparse_indexer_loss as npu

    from fsdp_turbo.ops.indexer_kl_metric import collect_indexer_kl_metrics

    metric_impl = npu.indexer_kl_metric

    def checked_metric(target, probs, *, use_fused):
        assert use_fused == use_fused_kl
        return metric_impl(target, probs, use_fused=use_fused)

    monkeypatch.setattr(npu, "indexer_kl_metric", checked_metric)
    observed = []
    metadata_widths = []

    def metadata(*args, **kwargs):
        metadata_widths.append(kwargs["topk"])

    def native_grad(q, k, w, indices, teacher, **kwargs):
        assert indices.shape == teacher.shape == (1, 4, 1, 2)
        assert indices.dtype == torch.int32
        observed.append(teacher.detach().clone())
        probs = teacher
        if layout == "bsk":
            probs = teacher.squeeze(2)
        elif layout == "swapped":
            probs = teacher.transpose(1, 3)
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(w), probs

    fake_ops = SimpleNamespace(
        sparse_lightning_indexer_kl_loss_grad_metadata=metadata,
        sparse_lightning_indexer_kl_loss_grad=native_grad,
    )
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", SimpleNamespace(ops=fake_ops))
    q = torch.randn(1, 4, 32, 128, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 4, 128, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(1, 4, 32, requires_grad=True)
    indices = torch.tensor([[[0, -1], [0, 1], [1, 2], [-1, -1]]], dtype=torch.int32)
    teacher = torch.tensor([[[3.0, 0.0], [2.0, 6.0], [10.0, 10.0], [0.0, 0.0]]])
    expected = torch.tensor([[[1.0, 0.0], [0.25, 0.75], [0.5, 0.5], [0.0, 0.0]]])
    for mass in (1.0, 17.0):
        with collect_indexer_kl_metrics(use_fused=use_fused_kl) as metrics:
            carrier = npu.sparse_indexer_gradient_npu(q, k, w, indices, teacher * mass, 1)
        grads = torch.autograd.grad(carrier * 0.25, (q, k, w))
        for grad in grads:
            torch.testing.assert_close(grad, torch.full_like(grad, 0.25))
        (metric,) = metrics.sums
        if layout == "swapped":
            assert metric.isnan()
        else:
            torch.testing.assert_close(metric, torch.zeros_like(metric), atol=1e-6, rtol=0)
        actual = observed[-1].squeeze(2)
        torch.testing.assert_close(actual, expected)
        assert metadata_widths[-1] == 2
        assert actual.dtype == torch.float32


def test_cpu_attention_dispatch_on_npu_host(monkeypatch):
    from types import SimpleNamespace
    from fsdp_turbo.ops.sparse_attention import indexed_sparse_attention
    from fsdp_turbo.ops.cpu.sparse_attention import indexed_sparse_attention_torch

    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda *args: SimpleNamespace(type="npu"))
    q = torch.randn(1, 2, 2, 4)
    k = torch.randn(1, 2, 4)
    sinks = torch.zeros(2)
    indices = torch.tensor([[[0], [1]]], dtype=torch.int32)
    actual = indexed_sparse_attention(q, k, sinks, indices, 0.5, use_sparse_flash_attn=True)
    expected = indexed_sparse_attention_torch(q, k, sinks, indices, 0.5)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("fused", [False, True])
def test_tnd_adapter_requires_sfa_before_backbone(monkeypatch, fused):
    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import build_deepseek_v41_model

    model = build_deepseek_v41_model(
        model_args=_adapter_args(),
        use_sparse_flash_attn=False,
        use_fused_lightning_indexer_loss=fused,
        indexer_loss_coeff=0.1,
    )

    def reject_backbone(*args, **kwargs):
        pytest.fail("invalid TND backend must be rejected before the backbone")

    monkeypatch.setattr(model.model, "forward_backbone", reject_backbone)
    with pytest.raises(ValueError, match="requires use_sparse_flash_attn=True"):
        model(
            torch.ones(1, 10, dtype=torch.long),
            sfa_layout="TND",
            actual_seq_len=torch.tensor([4, 10], dtype=torch.int32),
        )


@pytest.mark.parametrize("use_fused_kl", [False, True])
def test_trainer_reports_kl_only_on_logging_step(use_fused_kl):
    from types import SimpleNamespace
    from fsdp_turbo.models.deepseek.deepseek_v41.adapter import build_deepseek_v41_model
    from fsdp_turbo.training.trainer import BaseTrainer

    model = build_deepseek_v41_model(model_args=_adapter_args(), indexer_loss_coeff=0.3)
    torch.manual_seed(51)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.2)
    ids = torch.randint(0, 32, (2, 10))
    batch = dict(input_ids=ids, labels=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))
    batch["attention_mask"][1, -2:] = False
    trainer = SimpleNamespace(
        model=model,
        _global_step=0,
        _accumulated_losses={},
        config=SimpleNamespace(
            model=SimpleNamespace(track_indexer_kl_loss=True, use_fused_indexer_kl_loss=use_fused_kl),
            run=SimpleNamespace(logging_steps=2, gradient_accumulation_steps=1),
        ),
    )
    plain, _ = BaseTrainer.train_step(trainer, batch)
    assert "indexer_kl" not in trainer._accumulated_losses
    reference_grad = model.model.layers[0].attn.indexer.wq_b.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    trainer._global_step = 1
    tracked, _ = BaseTrainer.train_step(trainer, batch)
    (metric,) = trainer._accumulated_losses["indexer_kl"]
    assert metric.item() > 0 and not metric.requires_grad
    torch.testing.assert_close(tracked, plain)
    torch.testing.assert_close(model.model.layers[0].attn.indexer.wq_b.weight.grad, reference_grad)
