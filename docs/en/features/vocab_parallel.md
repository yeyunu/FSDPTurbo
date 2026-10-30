# Vocabulary Parallelism

FSDPTurbo provides vocabulary parallelism (VP) for very large vocabulary embedding tables (such as the PLE n-gram table of Qwen4-Exp, with about 51B parameters). It shards the `[vocab, dim]` weight of `nn.Embedding` along the vocabulary dimension (dim 0) across multiple devices, and token IDs are dispatched through All-to-All communication to the devices holding the corresponding vocabulary shards for embedding lookup, so that a very large embedding table can **reside on the device and remain trainable**. VP is the counterpart of expert parallelism (EP) on embedding tables: experts are sharded along the expert dimension, while embedding tables are sharded along the vocabulary dimension.

## Configuration

Configure the vocabulary parallelism degree and the VP plan in the `distributed` section (using the Qwen4-Exp PLE n-gram table as an example):

```yaml
distributed:
  vocab_parallel_size: 8
  vocab_fully_shard_parallel_size: 1
  vocab_data_parallel_size: 1
  vp_plan:
    apply_modules:
      - model.language_model.layers.{*}.ple.ple_embedding.ngram_embedding
```

For a complete training example, see `examples/qwen3_8`.

## Parameter Descriptions

| Parameter | Description |
|---|---|
| `vocab_parallel_size` | Vocabulary parallel degree; the embedding table is sharded along the vocabulary dimension across the VP group |
| `vocab_fully_shard_parallel_size` | Vocabulary FSDP2 (V-FSDP) parallel degree; further fully shards the VP shards. When not set, it is automatically computed as `world_size // (vocab_parallel_size × vocab_data_parallel_size)` |
| `vocab_data_parallel_size` | Vocabulary data parallel degree |
| `vp_plan.apply_modules` | List of modules to which VP is applied, supporting the `model.language_model.layers.{*}` wildcard; the matched target must be `nn.Embedding` |
| `vp_plan.apply_vfsdp_modules` | List of modules to which V-FSDP is applied; defaults to the same as `apply_modules` |
| `vp_plan.gradient_divide_factor` | Gradient divide factor of the backward All-to-All; defaults to `WORLD_SIZE` (consistent with the gradient averaging semantics of FSDP for dense parameters) |

## How It Works

1. **Weight sharding**: `vocab_parallelize_modules` shards the matched embedding weight along the vocabulary dimension into a DTensor with `Shard(0)`, and records the vocabulary range `[start, end)` of each rank. For tables that reside on the host (such as the PLE table shared through `/dev/shm`), only this rank's shard is moved onto the accelerator, avoiding OOM caused by moving the whole table onto the device.
2. **All-to-All embedding lookup**: the module's forward is replaced with an all-to-all dispatcher: sort and group by the rank that owns each token ID → All-to-All dispatch of the IDs → each rank performs the lookup with `F.embedding` on its local shard → a differentiable All-to-All returns the embedding vectors → an inverse permutation restores the input order. The VP group also acts as a data parallel group (each rank embeds different tokens).
3. **V-FSDP second-level sharding**: `vocab_fully_shard_modules` applies FSDP2 fully sharding on top of the VP shards, further reducing the per-card storage to `[vocab/(vp×vfsdp), dim]`; the forward unshards and only reassembles this rank's contiguous vocabulary window, and no rank ever materializes the whole table.
4. **Pure VP mode**: when `vocab_data_parallel_size × vocab_fully_shard_parallel_size == 1`, fully_shard is skipped, the VP parameters stay outside the FSDP unit, and the mixed precision conversion and gradient division are instead performed in the backward All-to-All.

VP uses an independent `vdp/vfsdp/vp` mesh and can coexist with expert parallelism. `FSDPTurbo` applies VP before applying FSDP: V-FSDP modules are taken over by `vp_plan` (the related patterns are automatically removed from `fsdp_plan.apply_modules`), and pure VP modules are automatically added to `fsdp_plan.ignored_modules`; in neither case is manual adjustment of the FSDP plan required.

## Module Requirements

- The matched target must be `nn.Embedding`, and the vocabulary size `num_embeddings` must be divisible by `vocab_parallel_size`.
- The `max_norm`, `scale_grad_by_freq`, and `sparse` options of `nn.Embedding` are not supported.

## Notes

- **Enabling conditions**: configuring `vp_plan` requires `vocab_parallel_size > 1`, otherwise an error is raised. When `vp_plan` is not configured, a very large embedding table can fall back to the frozen CPU offload path (such as `ple_embedding_on_cpu`, where the table is not trainable).
- **Parallelism combination constraint**: `vocab_data_parallel_size × vocab_fully_shard_parallel_size × vocab_parallel_size` must equal the total number of devices.
- **Relationship with FSDP**: VP modules cannot be taken over by an ancestor FSDP unit; the configuration validation handles pattern ownership automatically, with no manual adjustment required.
- **Memory estimation**: taking the PLE table with ~51B parameters (bf16) as an example, with `vp=8, vfsdp=1` each card holds about 12.8 GB of parameters and the corresponding optimizer states.
