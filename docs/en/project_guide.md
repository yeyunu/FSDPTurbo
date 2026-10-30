# FSDPTurbo Project Guide

Welcome to contribute code and documentation to FSDPTurbo!

## Project Introduction

The FSDPTurbo project code is organized according to modular design principles and mainly contains the following core modules:

- `fsdp_turbo/`: core code directory, containing implementations of core features such as distributed parallelism strategies, memory optimization, quantization training, and the training framework
- `examples/`: model example scripts, covering training examples for models such as Qwen, Kimi, and DeepSeek
- `tests/`: test cases directory, containing unit tests and system tests
- `docs/`: project documentation directory, providing the installation guide, quick start, feature descriptions, and training solutions
- `ci/`: CI build and test scripts
- `pre-commit/`: code quality hook configuration, including spell checking and secret scanning

## Code Directory Structure

The modules listed above are organized as directories and files in the project repository. The following is an overview of the project directory structure:

```text
FSDPTurbo/
 ├── ci/                        # CI build and test scripts
 ├── docs/                      # Project documentation directory
 │   └── zh/                    # Chinese documentation
 ├── examples/                  # Model example scripts
 ├── fsdp_turbo/                # Core code directory
 ├── pre-commit/                # pre-commit hook configuration
 ├── tests/                     # Test cases directory
 ├── setup.py                   # Installation configuration file
 ├── pyproject.toml             # Project configuration file
 ├── README.md                  # Project documentation
 └── LICENSE                    # License file
```

Among these, `fsdp_turbo/` is the most core code directory, with the following internal structure:

```text
fsdp_turbo/
 ├── distributed/               # Distributed parallelism strategies
 │   ├── context_parallel/      # Context parallelism
 │   │   ├── kv_allgather/      # KV AllGather context parallelism
 │   │   └── ulysses/           # Ulysses context parallelism
 │   ├── expert_parallel/       # Expert parallelism (dispatcher and EFSDP)
 │   ├── fine_grained_fully_shard/ # Fine-grained fully sharding
 │   ├── fully_shard_parallel/  # FSDP2 fully sharded data parallelism
 │   ├── tensor_parallel/       # Tensor parallelism
 │   ├── parallel_state.py      # DeviceMesh and parallelism management
 │   └── dist_ops.py            # Distributed communication operations
 ├── memory/                    # Memory optimization
 │   ├── chunk_batch/           # Batch chunking
 │   ├── chunk_loss/            # Chunk loss computation
 │   ├── recompute/             # Activation recomputation
 │   ├── swap_activation/       # Activation swapping
 │   └── swap_optimizer/        # Optimizer state swapping
 ├── models/                    # Built-in model implementations
 │   ├── kimi/                  # Kimi K3 model
 │   └── qwen/                  # Qwen3.5 series models
 ├── modules/                   # General model components
 ├── ops/                       # Multi-backend operators
 │   ├── npu/                   # NPU backend implementation
 │   ├── cpu/                   # CPU implementation
 │   └── triton/                # Triton kernel implementation
 ├── quantization/              # Quantization training
 │   ├── converter/             # Module quantization converter
 │   ├── core/                  # Quantization core workflow
 │   ├── mx_formats/            # MX format operators
 │   └── qat/                   # Quantization-aware training
 ├── training/                  # Training framework
 │   ├── factories/             # Model/optimizer/scheduler/tokenizer factories
 │   ├── checkpoint.py          # Checkpoint save and restore
 │   ├── dataloader.py          # Data loading
 │   ├── trainer.py             # Main training loop
 │   ├── monitor.py             # Training monitoring
 │   └── profiler.py            # Performance data profiling
 ├── utils/                     # General utilities
 ├── fsdp_turbo.py              # Training entry point
 └── fsdp_turbo_config.py       # Full training configuration definition
```

## Core Submodule Descriptions

The directory tree above shows the overall structure of the project, where each submodule under `fsdp_turbo/` has its own responsibility:

- `distributed/`: parallelism strategy implementation. `parallel_state.py` manages the three-dimensional DeviceMesh and automatic derivation of parallelism degrees; `fully_shard_parallel/` provides FSDP2 fully sharded data parallelism, `tensor_parallel/` provides tensor parallelism, `expert_parallel/` provides expert parallelism (including dispatcher communication and EFSDP expert sharding), `context_parallel/` provides two context parallelism approaches — KV AllGather and Ulysses, and `fine_grained_fully_shard/` supports fine-grained parameter sharding.
- `memory/`: memory optimization techniques, including activation recomputation, activation swapping, optimizer state swapping, batch chunking, and chunk loss computation.
- `models/`: built-in model implementations (Qwen3.5, Qwen3.5-MoE, Kimi K3, etc.).
- `modules/`: general model components (RMSNorm, KDA, GatedDeltaNet, HyperConnection, short convolution, etc.).
- `ops/`: operator implementations organized by device backend (`npu/`, `cpu/`, `triton/`), uniformly registered via `registry.py` and automatically dispatched according to the running device.
- `quantization/`: quantization training, including MXFP8 format operators, W8A16 quantization-aware training, and the module quantization converter.
- `training/`: training lifecycle management, including the main training loop, checkpoint save/restore, data loading, training monitoring, performance data profiling, and model/optimizer/scheduler/tokenizer factories.
- `utils/`: general utilities for devices, precision, logging, patches, and more.

`fsdp_turbo.py` is the training entry point, and `fsdp_turbo_config.py` defines the full training configuration.

## Model Running Examples

The `examples/` directory provides complete training examples for mainstream models. Each example contains `config.yaml` (training configuration), `run.sh` (torchrun launch script), and `train.py` (training entry script):

```text
examples/
├── deepseekv4/     # DeepSeek V4 example
├── kimi_k3/        # Kimi K3 example
├── qwen3_5/        # Qwen3.5 example
├── qwen3_5_moe/    # Qwen3.5-MoE example
└── qwen3_moe/      # Qwen3-MoE example
```

Enter the corresponding model directory to launch training:

```bash
# Single node, 16 NPUs
NNODES=1 GPUS_PER_NODE=16 bash run.sh
# For multi-node, additionally specify the master node IP and port
NNODES=2 GPUS_PER_NODE=16 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash run.sh
```

For the complete usage workflow, see [Quick Start](./training/quick_start_fsdpturbo.md); for the list of supported models, see [Supported Models](./models/supported_models.md).

## Summary

FSDPTurbo provides a large-model distributed training solution that combines high performance with ease of use, with the following core features:

- **Multi-dimensional parallelism**: supports free combination of FSDP/TP/EP/EFSDP/CP multi-dimensional parallelism, selects modules based on wildcard patterns, decoupled from specific model structures
- **Easy integration**: built on PyTorch FSDP2, directly connected to the Hugging Face Transformers ecosystem; `FSDPTurbo(config, model)` completes the parallelization
- **Memory optimization**: provides activation recomputation, chunk loss, activation swapping, and optimizer state swapping, supporting large-model training with low memory usage
- **Quantization acceleration**: supports MXFP8 quantization formats, block-wise quantization, and quantization-aware training (QAT), compatible with the expert parallelism path
- **Multi-backend support**: built for Ascend NPUs; operators are uniformly registered by device and automatically dispatched
- **Out of the box**: provides training example scripts and configurations for mainstream models such as Qwen, Kimi, and DeepSeek

Through a clear directory structure, a comprehensive documentation system, and rich example scripts, the project helps developers efficiently carry out large-model distributed training.

## Contribution Process

1. Fork this repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Submit a Pull Request.

## Issue Feedback

- Submit functional issues or compliance issues via [Issues](https://gitcode.com/Ascend/FSDPTurbo/issues).
- If the documentation references your model or dataset and you do not wish to be mentioned, please submit an issue to let us know, and we will delete or update the relevant description as you request.
