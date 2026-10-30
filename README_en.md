# FSDPTurbo

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**A distributed training acceleration library with high performance and ease of use**

</div>

FSDPTurbo is a distributed training acceleration library that combines high performance with ease of use. Built on a design philosophy of high cohesion and low coupling, it delivers a general-purpose acceleration solution decoupled from the model architecture for training large-scale deep learning models.

FSDPTurbo is built on PyTorch FSDP2 (DTensor fully sharded data parallel) and directly connects to the Hugging Face Transformers ecosystem: after loading a model with the native `AutoModel`, users can complete the parallelization simply by calling `FSDPTurbo(config, model)`, significantly reducing the integration cost of distributed training for large models.

FSDPTurbo targets Ascend NPUs and additionally provides:

- **Multi-dimensional parallelism strategies**: free combination of FSDP2 fully sharded data parallelism, tensor parallelism (TP), expert parallelism (EP), and context parallelism (CP)
- **Memory optimization**: activation recomputation, batch chunking, chunk loss computation, activation/optimizer state swapping, and CPU offload
- **Quantization acceleration**: MXFP8 (E4M3/E5M2/HIF8) block-wise quantization, W8A16 quantization-aware training, and low-precision AllGather
- **Out-of-the-box model support**: training adaptations and scripts preconfigured for model series such as Qwen3.5, Qwen3 MoE, DeepSeek V4, and Kimi K3

For a detailed framework introduction, see [Project Introduction](docs/en/introduction.md).

# Latest News

---

- 🔥 **2026**: The FSDPTurbo repository officially released its first version, 0.1.0, providing multi-dimensional parallelism, memory optimization, and quantization training capabilities, with preconfigured training examples for models such as Qwen3.5, Kimi-K3, and DeepSeek-V4

# Directory Structure

---

The FSDPTurbo project code is organized according to modular design principles.

```shell
FSDPTurbo/
 ├── .gitcode                  # Gitcode platform configuration (PR templates, etc.)
 ├── ci                        # CI gate and Docker image builds
 ├── docs                      # Project documentation directory
 │   └── zh                    # Chinese documentation
 ├── examples                  # Training examples (organized by model)
 ├── fsdp_turbo                # Core code directory
 │   ├── distributed           # Distributed parallelism (FSDP/EFSDP/TP/EP/CP/parallel_state)
 │   ├── memory                # Memory optimization (recompute/batch chunking/chunk loss/swapping)
 │   ├── quantization          # Quantization (MXFP8/W8A16/model conversion framework)
 │   ├── modules               # Custom modules (GDN/KDA/short convolution, etc.)
 │   ├── ops                   # Backend operators (unified registration and dispatch for npu/cpu/triton/cuda)
 │   ├── training              # Training utilities (trainer/checkpoint/factories)
 │   └── utils                 # Utility functions
 ├── pre-commit                # pre-commit hook configuration
 ├── tests                     # Test cases directory (ST/UT)
 ├── pyproject.toml            # Build configuration
 ├── setup.py                  # Installation configuration file
 └── README.md                 # Project documentation
```

# Documentation Guide

---

The FSDPTurbo documentation directory contains the following core content:

- **Project Introduction**: [Introduction](docs/en/introduction.md) —— framework positioning, architecture, and features
- **Environment Installation Guide**: [Software Installation](docs/en/training/install_guide.md) —— hardware compatibility, software version compatibility, and installation configuration
- **Quick Start**: [Quick Start](docs/en/training/quick_start_fsdpturbo.md) —— an introductory guide from environment installation to launching training
- **User Guide**: [Model Usage Guide](docs/en/training/user_guide.md) —— pretraining and fine-tuning training solutions
- **Feature Description**: [Supported Features](docs/en/features/supported_features.md) —— all training features with documentation links
- **Performance Tuning**: [Performance Tuning](docs/en/training/tuning/performance_tuning.md)
- **Development Guide**: [Feature Development Guide](docs/en/develop/feature_development.md), [Model Adaptation Guide](docs/en/develop/model_adaptation.md)
- **Testing**: [Test Case Contribution Guide](tests/README.md)
- **Docker Images**: [FSDPTurbo Runtime Docker Image Overview](ci/OVERVIEW.zh.md)

# Release Notes

---

See [Release Notes](./docs/en/release_notes_fsdpturbo.md) for the latest and historical releases of FSDPTurbo, including the software compatibility table for each version, version compatibility notes, and update notes for the current version.

# Installation

---

- For detailed installation steps and environment configuration, refer to [FSDPTurbo Software Installation](docs/en/training/install_guide.md).

Install from source:

```bash
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
pip install -e .
```

# Quick Start

---

Guides developers to quickly launch pretraining and fine-tuning tasks for large language models/multimodal models. For specific operations, refer to:

- [Quick Start (using Qwen3.5-27B as an example)](./docs/en/training/quick_start_fsdpturbo.md)

FSDPTurbo has preconfigured the following training example scripts (located in `examples/`):

| Model | Example Directory | Key Features |
| --- | --- | --- |
| Qwen3.5-27B (multimodal) | [examples/qwen3_5](examples/qwen3_5) | FSDP2 + TP + Ulysses CP + recompute |
| Qwen3.5-35B-A3B | [examples/qwen3_5_moe](examples/qwen3_5_moe) | Dense features + EP + EFSDP |
| Qwen3 MoE | [examples/qwen3_moe](examples/qwen3_moe) | EP + custom modeling |
| DeepSeek V4 | [examples/deepseekv4](examples/deepseekv4) | EP + custom modeling |
| Kimi K3 | [examples/kimi_k3](examples/kimi_k3) | EP + custom modeling |

```python
import torch
from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, FSDPPlanConfig, TPPlanConfig
from fsdp_turbo.fsdp_turbo import FSDPTurbo

# Create/load the model in the native way
model = create_your_model()

# Configure parallelism strategies and memory optimization
config = FSDPTurboConfig(
    fully_shard_parallel_size=8,
    fsdp_plan=FSDPPlanConfig(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    ),
    tensor_parallel_size=4,
    tp_plan=TPPlanConfig(
        colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
        rowwise_parallel=['*.o_proj'],
    ),
    recompute=True,
    recompute_plan=['model.layers.*'],
)

# Complete the parallelization in one line
model = FSDPTurbo(config, model)

# Training loop
for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

For more configurations such as expert parallelism and quantization, see [Model Usage Guide](./docs/en/training/user_guide.md).

Actual training tasks are organized around the "three-piece set" of a training entry (`train.py`), training configuration (`config.yaml`), and launch script (`run.sh`). Data is automatically loaded and tokenized at startup, with no offline preprocessing required. To switch datasets, register a custom dataset by referring to the [Model Adaptation Guide](./docs/en/develop/model_adaptation.md); some models (such as Kimi K3 and DeepSeek V4) require a custom modeling implementation. See the corresponding model directories under `examples/` for complete examples.

# Supported Models

---

FSDPTurbo is decoupled from the model architecture and theoretically supports any Hugging Face Transformers model based on the Transformer architecture, adapting to different model structures through wildcard plans, covering dense, MoE, and multimodal architectures. For the list of supported models, see [Supported Models](./docs/en/models/supported_models.md).

# Training Solutions and Features

---

FSDPTurbo provides pretraining and fine-tuning training solutions for large language models/multimodal models, supports free combination of multi-dimensional parallelism strategies, and preconfigures memory optimization and quantization acceleration features for Ascend hardware:

- **Fully Sharded Data Parallel (FSDP2)**: implements the PyTorch FSDP2 API, supporting mixed-precision training, forward/backward prefetch, and selective module sharding
- **Tensor Parallelism (TP)**: supports column parallelism, row parallelism, and sequence parallelism, integrated with DTensor
- **Expert Parallelism (EP)**: designed specifically for mixture-of-experts (MoE) models, supporting multiple dispatcher types (`eager`, `fused`, `mc2`, `domino`)
- **Context Parallelism (CP)**: Ulysses and KV AllGather styles, supporting long-sequence training
- **Memory Optimization**: gradient checkpointing, batch chunking, chunk loss computation, activation swapping, and optimizer state swapping
- **Quantization Training**: MXFP8 quantization (E4M3, E5M2, HIF8), block-wise quantization, W8A16 quantization-aware training, and low-precision AllGather
- **Checkpoint-Based Resumable Training**: saving/restoring weights and optimizer states based on PyTorch Distributed Checkpoint (DCP)

For the complete feature list and detailed descriptions, see [Supported Features](./docs/en/features/supported_features.md).

# Testing

---

FSDPTurbo provides a dual testing system of unit tests (UT) and end-to-end system tests (ST, including baseline comparison) to safeguard core capabilities such as multi-dimensional parallelism (FSDP/TP/EP/CP), memory optimization, and quantization acceleration. For details, see [Test Case Contribution Guide](tests/README.md).

# Version Maintenance Policy

---

FSDPTurbo versions go through the following five maintenance stages:

| **Status**            | **Duration** | **Description**                                                     |
| --------------------- | ------------ | ------------------------------------------------------------ |
| Planned               | 1–3 months   | Planned features                                             |
| Development           | 3 months     | Features under development                                   |
| Maintenance           | 6–12 months  | Merge all resolved issues and release versions. Different maintenance strategies apply to different FSDPTurbo versions: regular versions and long-term support versions have maintenance cycles of 6 months and 12 months, respectively |
| Unmaintained          | 0–3 months   | Merge all resolved issues; no dedicated maintenance staff and no version releases |
| End of Life (EOL)     | N/A          | The branch no longer accepts any modifications                |

# FAQs

---

For common issues encountered during basic repository usage, refer to [FSDPTurbo FAQs](./docs/en/FAQ.md). For issues not covered in the FAQ, you can look for similar issues in the repository's [issues list](https://gitcode.com/Ascend/FSDPTurbo/issues) or submit a new issue.

# Security Statement

---

FSDPTurbo is intended for research and development purposes only. When using it, please comply with applicable laws and regulations, as well as the License requirements of the models and datasets used. For instructions on system hardening, data security, runtime security, communication security, and vulnerability reporting, see [FSDPTurbo Security Statement](./docs/en/SECURITYNOTE.md).

# Disclaimer

---

## To FSDPTurbo Users

1. All content provided by FSDPTurbo is for your non-commercial use only.
2. With respect to the models and datasets involved in the FSDPTurbo test cases and example files, the platform uses them for functional testing only. Huawei does not provide any model weights or datasets. If you use these data for training, please pay special attention to complying with the License of the corresponding models and datasets. Huawei shall not bear any liability for infringement disputes arising from your use of these models and datasets.
3. If you find any problems while using FSDPTurbo (including but not limited to functional and compliance issues), please submit an issue on GitCode, and we will review and resolve it promptly.
4. The third-party open-source software that FSDPTurbo features depend on is provided and maintained by the respective third-party communities. Fixes for issues caused by third-party open-source software rely on contributions and feedback from the relevant communities. You should understand that the FSDPTurbo repository does not guarantee fixes for issues in the third-party open-source software itself, nor does it guarantee that all vulnerabilities and errors of third-party open-source software will be tested or corrected.

## To Data Owners

If you do not want your model or dataset to be mentioned in FSDPTurbo, or wish to update the descriptions related to you in FSDPTurbo, please submit an issue on Gitcode, and we will delete or update the relevant descriptions as required by your issue. We sincerely appreciate your understanding of and contributions to FSDPTurbo.

# License

---

- For the license governing the use of the FSDPTurbo product, see [LICENSE](LICENSE).
- For files involved in FSDPTurbo: if a License exists in the file directory, that License prevails; if no License exists in the file directory, the Apache 2.0 license applies.

# Contribution Guide

---

If you wish to report issues or contribute code to FSDPTurbo, see the [Contribution Guide](./CONTRIBUTING.md).

# Acknowledgments

---

We thank every PR from the community and welcome contributions to FSDPTurbo.
