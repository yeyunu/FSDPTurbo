# Supported Models

FSDPTurbo is decoupled from the model architecture and theoretically supports all Hugging Face Transformers models based on the Transformer architecture, adapting to different model structures through wildcard plans. Based on model architecture characteristics, models are divided into two major categories: dense models and MoE models. See the support list below for details.

**Table field description**

| Field | Description |
| :--- | :--- |
| **Model** | Model name. |
| **Download Link** | Download address for the model weights; click to access the Hugging Face model repository directly. |
| **Example Directory** | Path of the model's training example in this project, for quickly locating and using the model. |
| **Sequence Length** | Maximum text sequence length configured in the training example. |
| **Cluster Size** | Cluster size configuration used by the training example, in the format of "number of nodes × number of cards". |
| **Key Features** | Combination of key features enabled in the training example. |

> [!NOTE]
>
> - The repository supports <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 training products</term>, and <term>Atlas A2 training products</term>; NPU product information can be obtained via [Hardware Information Query](https://www.hiascend.com/cann/download). In addition, the on-chip memory of a single NPU is required to be 64 GB or larger; after [installing the driver and firmware](../training/install_guide.md), the on-chip memory capacity of a single NPU can be checked with `npu-smi info`.
> - The default cluster size of each training example is shown in the "Cluster Size" column of the table above and is set via `NNODES` and `GPUS_PER_NODE` in `examples/*/run.sh`, where `NNODES` refers to the number of machines and `GPUS_PER_NODE` refers to the number of cards on each machine. If the actual number of cards differs, adjust the parallelism parameters under `distributed` accordingly.
> - If the Hugging Face community cannot be accessed smoothly to download weight resources, it is recommended to download weights from [ModelScope](https://www.modelscope.cn/models), and pay attention to the correctness and security of the files to be downloaded.

## Dense Models

A dense model is a traditional deep learning model structure in which the connections between neurons are dense: most or all neurons in each layer are connected to all neurons in the next layer.

| Model | Download Link | Example Directory | Sequence Length | Cluster Size | Key Features |
|---|---|---|---|---|---|
| Qwen3.5-27B (multimodal) | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | `examples/qwen3_5` | 2048 | 1x16 | FSDP2 + TP + Ulysses CP + Recomputation |

## MoE Models

An MoE model (Mixture of Experts Model) routes inputs to a subset of expert networks through a gating network, expanding the parameter scale while controlling computational cost.

| Model | Download Link | Example Directory | Sequence Length | Cluster Size | Key Features |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `examples/qwen3_5_moe` | 2048 | 1x16 | Dense features + EP (Expert Parallelism) + EFSDP |
| Qwen3 MoE | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `examples/qwen3_moe` | 4096 | 1x8 | EP + custom modeling |
| DeepSeek V4 | [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | `examples/deepseekv4` | 1024 | 1x8 | EP + custom modeling |
| Kimi K3 | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | `examples/kimi_k3` | 2048 | 1x16 | EP + custom modeling |

## Adapting New Models

See [Model Adaptation Guide](../develop/model_adaptation.md) for how to integrate new models.
