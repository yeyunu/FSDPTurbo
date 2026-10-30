# Chunk Loss Computation

Chunk loss computation processes the loss computation in pipelined chunks, avoiding the memory peaks caused by fully materializing logits with large-vocabulary models.

## How It Works

With large-vocabulary models (e.g., vocab_size 150k+), the logits output by `lm_head` have the shape `[batch, seq_len, vocab_size]`, and fully materializing them consumes a large amount of memory. Chunk loss computation splits the hidden states along the sequence dimension by `chunk_size`, computes the logits and loss chunk by chunk through `lm_head`, computes and accumulates gradients chunk by chunk with `torch.func.grad_and_value`, and finally aggregates them into the total loss.

## Configuration

Chunk loss computation is currently enabled through the model attribute `enable_chunk_loss` and is not yet included in the unified `config.yaml` configuration. To enable it:

```python
# Set it in the model definition
model.enable_chunk_loss = True
```

Once enabled, the model's forward internally forces `output_hidden_states=True`, and passes the last layer's hidden states, the `lm_head` weights, the loss computation function, and `chunk_size` to `chunk_loss()` to complete the chunked computation.

## Relationship with Context Parallelism (CP)

Chunk loss computation is **incompatible with CP and cannot be enabled at the same time**. If `enable_chunk_loss` is also set after Ulysses CP is enabled, the model raises an error directly (see `fsdp_turbo/models/kimi/kimi_k3_cp.py`):

```text
ValueError: Kimi K3 chunk loss is not supported together with Ulysses CP.
```

Although both relieve memory pressure in large-vocabulary/long-sequence scenarios, their paths conflict:

- CP's `loss_function_patches` is responsible for global label shift alignment and cross-rank loss reduction, but it still materializes the full local logits on the local rank.
- Chunk loss computation avoids fully materializing logits by computing chunk by chunk, and needs to take over the entire hidden states → lm_head → loss computation flow.
- The two have mutually exclusive entry points for taking over the loss computation, and the current implementation does not support combining them.

## Applicable Scenarios

- Large-vocabulary models (vocab_size > 100k)
- Ultra-long-sequence training (seq_len > 32k)
- Scenarios with tight memory that cannot afford fully materializing logits
