# Supported Features

This table summarizes the training features supported by FSDPTurbo, grouped by feature category. See the corresponding feature documents for details.

<table>
  <thead>
    <tr>
      <th>Feature Category</th>
      <th>Feature Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">Parallel Features</td>
      <td><a href="./fully_shard_parallel.md">Fully Sharded Data Parallel (FSDP2)</a></td>
      <td>Based on PyTorch FSDP2, fully shards parameters/gradients/optimizer states, supports mixed precision and prefetch</td>
    </tr>
    <tr>
      <td><a href="./tensor_parallel.md">Tensor parallelism</a></td>
      <td>Column parallelism, row parallelism, sequence parallelism</td>
    </tr>
    <tr>
      <td><a href="./expert_parallel.md">Expert parallelism</a></td>
      <td>MoE expert parallelism, supports eager/fused/mc2/domino dispatchers and EFSDP</td>
    </tr>
    <tr>
      <td><a href="./context_parallel.md">Context parallelism</a></td>
      <td>Ulysses context parallelism, supports long-sequence training</td>
    </tr>
    <tr>
      <td><a href="./vocab_parallel.md">Vocabulary parallelism</a></td>
      <td>Splits the embedding table of extremely large vocabularies along the vocabulary dimension (VP/V-FSDP), with All-to-All lookup, resident on device and trainable</td>
    </tr>
    <tr>
      <td rowspan="7">Memory Optimization Features</td>
      <td><a href="./recompute.md">Activation recomputation</a></td>
      <td>Selective activation recomputation, reducing activation memory usage</td>
    </tr>
    <tr>
      <td><a href="./chunk_batch.md">Batch chunking</a></td>
      <td>Splits module forward along the batch dimension, reducing per-layer memory peaks</td>
    </tr>
    <tr>
      <td><a href="./chunk_loss.md">Chunk loss computation</a></td>
      <td>Chunk loss computation, suitable for large-vocabulary models</td>
    </tr>
    <tr>
      <td><a href="./swap_activation.md">Activation swapping</a></td>
      <td>Asynchronously offloads activations to CPU memory, enabled via YAML configuration</td>
    </tr>
    <tr>
      <td><a href="./swap_optimizer.md">Optimizer state swapping</a></td>
      <td>Offloads optimizer states to CPU (<code>optimizer_type: AdamWSwap</code>)</td>
    </tr>
    <tr>
      <td><a href="./fully_shard_parallel.md">CPU offload</a></td>
      <td>FSDP2 parameter/gradient CPU offload</td>
    </tr>
    <tr>
      <td><a href="./checkpoint_resume.md">Checkpoint-based resumable training</a></td>
      <td>Save/restore of weights and optimizer states</td>
    </tr>
    <tr>
      <td rowspan="2">Quantization Features</td>
      <td><a href="./quantization.md">MXFP8 quantization training</a></td>
      <td>Supports E4M3/E5M2/HIF8 formats and block-wise quantization</td>
    </tr>
    <tr>
      <td><a href="./quantization.md">Low-precision AllGather</a></td>
      <td>Low-precision FSDP2 communication</td>
    </tr>
    <tr>
      <td rowspan="2">Others</td>
      <td><a href="./memory_snapshot.md">Memory snapshot</a></td>
      <td>Collects the allocator history and snapshots for memory analysis, enabled via YAML configuration, supports automatic dumping on OOM</td>
    </tr>
    <tr>
      <td><a href="./environment_variable.md">Environment variables</a></td>
      <td>Environment variables for cluster configuration and NPU performance tuning</td>
    </tr>
  </tbody>
</table>
