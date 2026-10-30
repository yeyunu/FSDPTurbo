# FSDPTurbo Documentation Guide

## Documentation Introduction

FSDPTurbo documentation is organized by content type under `docs/en/` and mainly includes the following core directories:

- `training/`: training guides, including installation, quick start, pretraining, fine-tuning, and performance tuning
- `features/`: feature descriptions, including parallelism strategies, memory optimization, quantization, and checkpoint-based resumable training
- `develop/`: development guides, including model adaptation and feature development
- `models/`: supported model lists
- `tools/`: development tools, including performance data profiling

## Documentation Directory Structure

The hierarchy of the FSDPTurbo documentation directory is described below:

```text
docs/en/
├── introduction.md           # Product introduction
├── project_guide.md          # Project guide
├── docs_guide.md             # Documentation guide
├── appendixes.md             # Appendix (glossary, etc.)
├── release_notes_fsdpturbo.md # Release notes
├── SECURITYNOTE.md           # Security statement
├── FAQ.md                    # FAQs
├── develop/                  # Development guide
│   ├── model_adaptation.md   # Model adaptation guide
│   └── feature_development.md # Feature development guide
├── features/                 # Feature descriptions
│   ├── supported_features.md # Feature list
│   ├── fully_shard_parallel.md # FSDP2 fully sharded data parallelism
│   ├── tensor_parallel.md    # Tensor parallelism
│   ├── expert_parallel.md    # Expert parallelism
│   ├── context_parallel.md   # Context parallelism
│   ├── quantization.md       # Quantization training
│   ├── recompute.md          # Activation recomputation
│   ├── chunk_batch.md        # Batch chunking
│   ├── chunk_loss.md         # Chunk loss computation
│   ├── swap_activation.md    # Activation swapping
│   ├── swap_optimizer.md     # Optimizer state swapping
│   ├── checkpoint_resume.md  # Checkpoint-based resumable training
│   └── environment_variable.md # Environment variable descriptions
├── models/                   # Supported model lists
│   └── supported_models.md
├── tools/                    # Development tools
│   └── profiling.md          # Performance data profiling
└── training/                 # Training guides
    ├── install_guide.md      # Software installation
    ├── quick_start_fsdpturbo.md # Quick start
    ├── user_guide.md         # Model usage guide
    ├── pretrain/             # Pretraining
    │   └── pretrain.md       # Distributed training of large models
    ├── finetune/             # Fine-tuning
    │   └── finetune.md       # Model fine-tuning
    └── tuning/               # Performance tuning
        └── performance_tuning.md # Model performance tuning
```

## Core Documentation Guide

Quick jump: [Getting Started](#getting-started) | [Training Solutions](#training-solutions) | [Feature Descriptions](#feature-descriptions) | [Development Guide](#development-guide) | [Tools and Tuning](#tools-and-tuning) | [Others](#others)

### Getting Started

| Content | Description |
| --- | --- |
| [FSDPTurbo Introduction](./introduction.md) | Product positioning and core capabilities |
| [Software Installation](./training/install_guide.md) | Environment dependencies and installation guide |
| [Quick Start](./training/quick_start_fsdpturbo.md) | Hands-on guide from environment installation to launching training |
| [Supported Models](./models/supported_models.md) | List of models supported by the repository |

### Training Solutions

| Category | Content | Description |
| --- | --- | --- |
| Guide | [Model Usage Guide](./training/user_guide.md) | Guide for choosing solutions and configurations based on training needs |
| Pretraining | [Distributed Training of Large Models](./training/pretrain/pretrain.md) | Distributed pretraining solution |
| Fine-tuning | [Model Fine-tuning](./training/finetune/finetune.md) | Model fine-tuning solution |

### Feature Descriptions

| Category | Content | Description |
| --- | --- | --- |
| Overview | [Supported Features](./features/supported_features.md) | Overview of features supported by the repository |
| Parallelism strategies | [Fully Sharded Data Parallel (FSDP2)](./features/fully_shard_parallel.md) | FSDP2 parameter and gradient sharding |
| Parallelism strategies | [Tensor Parallelism](./features/tensor_parallel.md) | Intra-layer tensor parallelism |
| Parallelism strategies | [Expert Parallelism](./features/expert_parallel.md) | MoE expert parallelism |
| Parallelism strategies | [Context Parallelism](./features/context_parallel.md) | Context parallelism for long sequences |
| Memory and computation optimization | [Activation Recomputation](./features/recompute.md) | Discard and recompute activations to save memory |
| Memory and computation optimization | [Batch Chunking](./features/chunk_batch.md) | Further chunking of micro-batches |
| Memory and computation optimization | [Chunk Loss Computation](./features/chunk_loss.md) | Chunk loss computation to reduce peak memory usage |
| Memory and computation optimization | [Activation Swapping](./features/swap_activation.md) | Swapping activations with NPU memory |
| Memory and computation optimization | [Optimizer State Swapping](./features/swap_optimizer.md) | Swapping optimizer states with host memory |
| Others | [Quantization Training](./features/quantization.md) | Quantization training feature |
| Others | [Checkpoint-based Resumable Training](./features/checkpoint_resume.md) | Training resume via checkpoint save and restore |
| Others | [Environment Variables](./features/environment_variable.md) | Cluster and performance tuning environment variables |

### Development Guide

| Content | Description |
| --- | --- |
| [Model Adaptation Guide](./develop/model_adaptation.md) | Adaptation method for integrating new models into FSDPTurbo |
| [Feature Development Guide](./develop/feature_development.md) | Method for developing new features |

### Tools and Tuning

| Content | Description |
| --- | --- |
| [Performance Data Profiling](./tools/profiling.md) | Collecting profiling data based on Ascend chips |
| [Model Performance Tuning](./training/tuning/performance_tuning.md) | Performance and memory tuning guide |

### Others

| Content | Description |
| --- | --- |
| [FAQs](./FAQ.md) | Common issues and solutions |
| [Appendix](./appendixes.md) | Glossary, etc. |
| [Project Guide](./project_guide.md) | Project guide |
| [Release Notes](./release_notes_fsdpturbo.md) | Release notes |
| [Security Statement](./SECURITYNOTE.md) | Security usage notice |

## Documentation Writing Conventions

Core conventions for documentation contributors:

- Parameter names, default values, valid ranges, and effective conditions are governed by the code (see `fsdp_turbo/fsdp_turbo_config.py` for configuration definitions); capabilities not implemented in the code must not be documented.
- Use relative-path links between documents; images are stored under `docs/en/figures/<topic>/` and referenced with relative paths.
- Annotate code blocks with the language matching their content: `bash` for commands, `yaml` for configurations, `python` for examples; configuration examples must be consistent with `examples/*/config.yaml`.
- Follow the glossary in the [Appendix](./appendixes.md) for consistent terminology, and update it when introducing new terms.
- Use `> [!NOTE]` for supplementary explanations and `> [!NOTICE]` for security and risk warnings.
- After adding new documentation, evaluate and update the `_menu.md` navigation and this page's directory structure accordingly.
