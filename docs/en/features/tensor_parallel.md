# Tensor Parallelism

FSDPTurbo supports tensor parallelism (TP) based on DTensor, sharding the weight matrices within a single layer across multiple devices. It can be combined with other parallel strategies.

## Configuration

Configure the tensor parallel degree and the modules to shard in the `distributed` section:

```yaml
distributed:
  tensor_parallel_size: 4
  tp_plan:
    colwise_parallel: ['*.q_proj', '*.k_proj', '*.v_proj']   # Column-parallel modules
    rowwise_parallel: ['*.o_proj']                            # Row-parallel modules
    sequence_parallel: []                                     # Sequence-parallel modules
```

## Parameter Descriptions

| Parameter | Description |
|---|---|
| `tensor_parallel_size` | Tensor parallel degree |
| `colwise_parallel` | List of column-parallel modules; weights are sharded along the output dimension, supporting the `*.q_proj` wildcard |
| `rowwise_parallel` | List of row-parallel modules; weights are sharded along the input dimension |
| `sequence_parallel` | List of sequence-parallel modules, reducing activation memory when combined with TP |

## Combining with Other Parallel Strategies

- TP and FSDP2 can be freely combined: `tensor_parallel_size × fully_shard_parallel_size ≤ number of cards in a single node`.
- TP is applied before FSDP2: the modules are first sharded with TP, then the whole is fully sharded with FSDP2.

## Notes

- **Gradient synchronization**: the TP-sharded parameters are uniformly managed for gradient reduction by FSDP2 (TP column-parallel/row-parallel modules are no longer excluded from FSDP2 sharding); the precision can be aligned between enabling and disabling TP. Gradient norm computation recognizes both the `Shard` and `_StridedShard` placements.
- **Parallelism constraint**: `tensor_parallel_size` must evenly divide the number of attention heads and the hidden dimension; otherwise the column-parallel/row-parallel sharding will fail.
- **Recommended configuration**: within a single node, prefer the FSDP2 + TP combination (e.g., with 8 cards: `fully_shard_parallel_size=4, tensor_parallel_size=2`). TP should not be too large; ≤ 8 is recommended.
