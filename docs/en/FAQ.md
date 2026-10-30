# FAQs

This document collects common issues and solutions encountered while using FSDPTurbo, categorized by installation and environment, training startup, parallelism strategies, and precision and checkpoint-based resumable training.

## Installation and Environment

### What are the dependency version requirements?

Python >= 3.10 (3.11 or later recommended), PyTorch >= 2.9.0, transformers >= 5.2.0; the NPU environment additionally requires a matching version of torch_npu (same version as PyTorch) and triton-ascend >= 3.2.2. See [Software Installation](./training/install_guide.md) for the complete dependency list and verified version combinations.

### What if the NPU environment reports CANN/driver-related errors?

The CANN environment variables must be sourced in each new terminal before running training; without them, the business code cannot run:

```bash
# Default path for the root user
source /usr/local/Ascend/cann/set_env.sh
# Default path for non-root users
source ${HOME}/Ascend/cann/set_env.sh
```

### How do I verify that FSDPTurbo is installed successfully?

```bash
python -c "import fsdp_turbo; print(fsdp_turbo.__file__)"
```

In an NPU environment, additionally run `npu-smi info` to confirm the NPU status, and run the following command to confirm that torch_npu is available (output `True` means it is working normally):

```bash
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

## Training Startup

### How do I set up multi-node training?

After configuring the cluster parameters via environment variables, run `bash run.sh` on all nodes simultaneously: `MASTER_ADDR` (master node IP) and `MASTER_PORT` remain the same across nodes, while the node index (torchrun's `--node_rank`) differs per node:

```bash
# Execute on node 0
NNODES=2 GPUS_PER_NODE=16 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=0 bash run.sh
# Execute on node 1
NNODES=2 GPUS_PER_NODE=16 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=1 bash run.sh
```

When multi-node setup fails or initialization hangs, check the following in order:

- Whether `MASTER_PORT` is occupied; change the port and retry.
- Network connectivity between nodes and firewall port rules.
- For HCCL connection timeouts on large clusters, increase `HCCL_CONNECT_TIMEOUT` (recommended: `1800`).

See [Environment Variables](./features/environment_variable.md) for cluster variables and NPU performance tuning environment variables.

### What should I do when an out of memory (OOM) error occurs?

Handle it in the following order:

1. Reduce `data.batch_size` or `max_seq_length`; when the actual number of cards is smaller than the example default (16 cards), adjust the parallelism accordingly.
2. Enable activation recomputation with `memory.recompute: true` (the most commonly used and most reliable option).
3. If memory is still insufficient, additionally enable activation swapping with `memory.swap_activation: true` or `AdamWSwap` (prioritize when optimizer states occupy a lot of memory).
4. Enable `fsdp_plan.cpu_offload: true` when overall memory is tight.

See [Model Performance Tuning](./training/tuning/performance_tuning.md) for a comparison of the memory savings and time overhead of each approach.

### Where are training weights saved?

They are saved in the `checkpoint.output_dir` directory at intervals of `save_steps`. The saved contents include model parameters, optimizer states, learning rate scheduler states, and the training step, as detailed in [Checkpoint-Based Resumable Training](./features/checkpoint_resume.md).

## Parallelism Strategies

### Why doesn't the configured `data_parallel_size` take effect?

`data_parallel_size` and `expert_data_parallel_size` are auto-derived; configured values are overridden:

- `data_parallel_size` = total number of cards ÷ (`fully_shard_parallel_size` × `tensor_parallel_size`)
- `expert_data_parallel_size` = total number of cards ÷ (`expert_parallel_size` × `expert_fully_shard_parallel_size`)

Users only need to configure `fully_shard_parallel_size`, `tensor_parallel_size`, `ulysses_parallel_size`, `kv_allgather_parallel_size`, `expert_parallel_size`, and `expert_fully_shard_parallel_size`, and the divisibility constraint must be satisfied (the product of each combination must divide the total number of cards); otherwise initialization reports an error.

### Do I need to shard weights or convert formats before training?

No. FSDPTurbo loads Hugging Face-format weights directly and does not require conversion to formats such as Megatron; adjusting parallelism does not require re-sharding weights either — just configure the parallelism in the `distributed` section. Note that after loading weights, the training script performs a dtype conversion (`convert_model_dtype`) on model parameters and buffers according to the `torch_dtype` configuration; this is done automatically inside the script and requires no offline operations.

### What if an MoE model reports EP errors or the expert modules do not match?

Common causes and solutions:

- **Expert module structure requirement**: EP requires the experts module to hold 3D tensor parameters (`gate_up_proj`/`down_proj`, shape `[num_experts, out_dim, in_dim]`); independent expert submodules in ModuleList form must first be converted to a 3D tensor aggregation structure.
- **Divisibility constraint**: `expert_parallel_size` must divide the total number of experts per layer.
- **Dispatcher selection**: prefer `fused` / `mc2`; implement a custom EP forward when the expert computation semantics are non-standard.

See [Expert Parallelism](./features/expert_parallel.md) for details.

### What is `fixed_router` used for?

It is a debugging aid: when enabled, it replaces the model's actual routing output with fixed uniform routing (round-robin equal distribution) to avoid performance and precision issues caused by uneven load; it only takes effect when `expert_parallel_size > 1`.

## Precision and Checkpoint-based Resumable Training

### How do I troubleshoot precision anomalies?

- Confirm the precision configuration: `model.torch_dtype` (weight precision), `fsdp_plan.param_dtype` (forward computation precision, e.g. `bf16`), and `fsdp_plan.reduce_dtype` (gradient reduction precision, e.g. `fp32`).
- Enable `run.deterministic` when reproducible, deterministic results are needed.
- When gradient errors occur under EP and special parallelism combinations, check whether `ep_plan.gradient_divide_factor` needs to be specified manually.

### How do I use checkpoint-based resumable training?

Point `checkpoint.resume_from_checkpoint` to the corresponding checkpoint directory under `checkpoint.output_dir`; after training starts, each rank automatically loads its own shards and continues training from the saved step, with the learning rate and optimizer states restored synchronously. See [Checkpoint-Based Resumable Training](./features/checkpoint_resume.md) for details.

### Why does the step restart from 0 after resuming training?

When resuming, FSDPTurbo validates the dataset fingerprint (composed of `dataset_path`/`dataset_config`/`split`/`batch_size`): if it differs from the fingerprint at checkpoint saving time, `global_step` and `consumed_samples` are reset to 0 and a warning is printed to prevent training data from getting mixed up. Training resumes normally once the dataset configuration is confirmed unchanged.

### Can I resume training after changing the parallelism?

Weights are saved in the PyTorch Distributed Checkpoint (DCP) format, which supports loading under different parallelism degrees (parameters are re-sharded automatically), but the model structure must remain consistent; it is recommended to keep the parallelism configuration the same to avoid compatibility issues.
