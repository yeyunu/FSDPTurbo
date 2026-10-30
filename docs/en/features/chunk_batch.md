# Batch Chunking

Batch chunking splits the forward of the specified modules along the batch dimension into smaller micro-batches executed sequentially, then concatenates the outputs back to the original shape, reducing the memory peak of a single module's forward/backward pass.

## How It Works

When a module's activation values or temporary workspace peak scales with the batch size, slicing the input along the batch dimension can significantly reduce the memory peak. For example, with an input shape of `[4, 2048, hidden]` and `chunk_mbs=1` configured, the module's forward is executed 4 times with `[1, 2048, hidden]` inputs, and the outputs are concatenated back to `[4, 2048, hidden]`, transparent to the caller.

## Configuration

Enable it in the `memory` section of `config.yaml`:

```yaml
memory:
  chunk_batch: true
  chunk_batch_plan:
    chunk_mbs: 1
    apply_modules:
      - model.layers.{*}
    batch_dim: 0
    chunk_arg_indexes: [0]
    chunk_kwarg_names: ["hidden_states", "attention_mask"]
```

| Parameter | Description |
|---|---|
| `chunk_batch` | Master switch for batch chunking |
| `chunk_batch_plan.chunk_mbs` | Micro-batch size after chunking; default `1` |
| `chunk_batch_plan.apply_modules` | List of modules to which chunking is applied, supporting the `model.layers.{*}` wildcard |
| `chunk_batch_plan.batch_dim` | The tensor dimension representing the batch; usually `0` for Hugging Face decoder layers |
| `chunk_batch_plan.chunk_arg_indexes` | List of positional argument indexes to be chunked along the batch dimension |
| `chunk_batch_plan.chunk_kwarg_names` | List of keyword argument names to be chunked along the batch dimension |

> [!NOTE]
>
> When `chunk_batch` is enabled, `chunk_arg_indexes` or `chunk_kwarg_names` must be specified; otherwise a configuration validation error is raised at startup.

## Relationship with Context Parallelism (CP)

Batch chunking and CP are **incompatible and cannot be enabled at the same time**; the current implementation does not support combining the two.

Although both relieve memory pressure in large-batch/long-sequence scenarios, their chunking paths conflict with each other:

- When CP is enabled, the model backbone forward is replaced with a CP-aware implementation. The input is sharded along the **sequence dimension** to each CP rank before entering the decoder layers, the attention computation is also replaced with a CP-aware version that exchanges sequence/head-dimension information through cross-rank communication (such as Ulysses's all-to-all), and the global causal mask is reconstructed based on the local input of each rank.
- Batch chunking, on the other hand, wraps the forward of the decoder layers (see `fsdp_turbo/memory/chunk_batch/chunk_batch.py`), slicing the input along the **batch dimension** and executing it block by block, then concatenating the outputs back to the original shape.
- Both mechanisms take effect by taking over the forward entry. When stacked together, the shape conventions relied upon by sequence sharding, mask reconstruction, cross-rank communication, and output concatenation are no longer guaranteed, and the current implementation provides no combined adaptation.

> [!NOTICE]
>
> Unlike Chunk Loss (setting `enable_chunk_loss` after enabling CP directly raises an error), enabling batch chunking and CP at the same time has **no explicit validation at startup or in the forward pass** and will not raise an error immediately, but the behavior is undefined. Be sure to avoid this configuration combination.

## Applicable Scenarios

- Excessive single-layer forward memory peak during large-batch-size training
- Module temporary workspace scaling with the batch size (such as large-vocabulary projections, MoE grouped computation)
- Combined use with recomputation, activation swapping, and other techniques
