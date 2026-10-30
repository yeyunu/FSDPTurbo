# Context Parallelism

FSDPTurbo supports Ulysses-style context parallelism (CP), sharding the sequence dimension across multiple devices to enable ultra-long sequence training.

## Configuration

Configure the context parallel degree and function replacements in the `distributed` section:

```yaml
distributed:
  ulysses_parallel_size: 2
  cp_plan:
    ulysses_function_patches:
      - target_functions:
          - transformers.models.qwen3_5.modeling_qwen3_5.eager_attention_forward
        type: full_attention
      - target_functions:
          - transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward
        type: gated_delta_net
    loss_function_patches:
      - target_functions:
          - transformers.loss.loss_utils.ForCausalLMLoss
        type: causal_lm_loss
```

## Parameter Descriptions

| Parameter | Description |
|---|---|
| `ulysses_parallel_size` | Ulysses context parallel degree |
| `kv_allgather_parallel_size` | KV AllGather context parallel degree |
| `ulysses_function_patches` | List of attention function replacements: `target_functions` specifies the functions to be replaced (fully qualified names), and `type` specifies the replacement type (`full_attention`, `gated_delta_net`, etc.; Callable custom attention is also supported) |
| `kv_allgather_function_patches` | Function replacements for the KV AllGather scenario |
| `loss_function_patches` | Loss function replacements (such as `causal_lm_loss`), ensuring correct loss synchronization after CP sharding |

## Working with module_patches

Ulysses CP usually requires replacing the forward of the model backbone (to change the attention invocation and the sequence sharding approach):

```yaml
module_patches:
  - target: transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model.forward
    replacement: fsdp_turbo.models.qwen.qwen3_5.qwen3_5_model_forward
```

FSDPTurbo provides replacement implementations for several mainstream models (see `fsdp_turbo/models/`).

## CP Loss Synchronization Mechanism

CP shards the sequence across multiple ranks, with each rank holding only a partial sequence segment. To ensure correct loss computation:

1. **Global label shift**: before loss computation, the label shift `shifted_labels[..., :-1] = labels[..., 1:]` must be performed on the full sequence, and the shifted labels are then sharded to each CP rank, preserving the training samples across rank boundaries.
2. **Loss function replacement**: through `loss_function_patches`, the native loss function (such as `ForCausalLMLoss`) is replaced with a CP-aware version, avoiding repeated in-sequence shifts.
3. **Cross-rank synchronization**: the loss within the CP group is already synchronized in the forward pass, so no additional all-reduce is required.

## Notes

- **Sequence length alignment**: `max_seq_length` should be divisible by `ulysses_parallel_size` to avoid unequal-length segments during sequence sharding.
- **Data loading**: CP does not change the data loading approach; each DP rank still loads full samples, and the sequence sharding is done inside the model forward.
- **Mutually exclusive with Chunk Loss**: after enabling Ulysses CP, `enable_chunk_loss` cannot be set at the same time; the model will directly raise an error. See [Chunk Loss Computation](./chunk_loss.md) for details.
- **Mutually exclusive with Chunk Batch**: after enabling CP, `memory.chunk_batch` cannot be configured at the same time; this combination has no explicit error guard and its behavior is undefined. See [Batch Chunking](./chunk_batch.md) for details.
- **Typical long-sequence configuration**: for 32k+ long-sequence training, `ulysses_parallel_size=2~4` is recommended, combined with recomputation and FSDP2.
