# Quick Start: Pretraining and Fine-Tuning the Qwen3.5 Model

## Overview

This document provides a simple example to help developers new to FSDPTurbo quickly start model training tasks and complete fine-tuning based on a pretrained model. Using the Qwen3.5-27B model as an example, it guides developers through the pretraining and fine-tuning of a large language model. The main steps include:

- **Environment preparation**: Install FSDPTurbo and its dependencies
- **Model and dataset preparation**: Obtain the Qwen3.5-27B model weights and training data
- **Start a training task**: Run model pretraining and fine-tuning on NPU

> [!NOTE]
>
> - FSDPTurbo supports <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 training products</term>, and <term>Atlas A2 training products</term>. The per-device memory must meet the model size requirements. For details, see [Supported Models](../models/supported_models.md).
>
> - This document is a **step-by-step quick start tutorial**. To switch to a different model or register a custom dataset, see the [Model Adaptation Guide](../develop/model_adaptation.md).
>
> - In the current Qwen3.5-27B example script, `GPUS_PER_NODE=16` means 16 devices are required. If your actual setup has fewer devices, adjust the parallelism parameters accordingly; otherwise, you may encounter OOM (Out Of Memory) issues.

Prerequisites for developers:

- Basic experience with PyTorch
- Entry-level Python development experience
- Basic understanding of Hugging Face Transformers

## Environment Preparation

### Environment Setup

For environment setup, see [Software Installation](./install_guide.md).

### Model Weight Preparation

FSDPTurbo directly loads Hugging Face-format weights — **no weight format conversion is required**. After loading, the training script converts the dtype of model parameters and buffers according to the `torch_dtype` configuration. To obtain the Qwen3.5-27B weights:

1. Create a directory to store the weight files.

   ```shell
   mkdir -p ./model_from_hf/qwen3_5_hf
   cd ./model_from_hf/qwen3_5_hf
   ```

2. Obtain the model weight files from Hugging Face or ModelScope (choose one).

   Option 1: Obtain from Hugging Face.

   ```shell
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/config.json
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/generation_config.json
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/merges.txt
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/model.safetensors.index.json
   # Download all safetensors files according to the weight shard list in model.safetensors.index.json
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/model-00001-of-0000X.safetensors
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/tokenizer.json
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/tokenizer_config.json
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/vocab.json
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/preprocessor_config.json
   wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/chat_template.json
   ```

   Option 2: Obtain from ModelScope (recommended in China).

   ```shell
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/config.json
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/generation_config.json
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/merges.txt
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/model.safetensors.index.json
   # Download all safetensors files according to the weight shard list in model.safetensors.index.json
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/model-00001-of-0000X.safetensors
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/tokenizer.json
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/tokenizer_config.json
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/vocab.json
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/preprocessor_config.json
   wget https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/resolve/master/chat_template.json
   ```

   > [!NOTE]
   >
   > You can also directly set `"Qwen/Qwen3.5-27B"` in the training configuration, and the framework will download it automatically.

3. Use sha256sum to verify the correctness and integrity of the weight files.

   ```shell
   # Open the file details to get the sha256 value, https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/model-00001-of-0000X.safetensors
   # or https://www.modelscope.cn/models/Qwen/Qwen3.5-27B/file/view/master/model-00001-of-0000X.safetensors
   sha256sum ./model-00001-of-0000X.safetensors
   # Run the same verification on the remaining weight shard files
   cd ../..
   ```

The complete model directory should contain the model configuration, tokenizer configuration, and all weight files.

### Dataset Preparation

FSDPTurbo data loading is implemented based on Hugging Face `datasets` and supports local data files or Hugging Face Hub datasets. Loading and tokenization are completed when the training task starts; no prior conversion to Megatron Indexed Dataset is required.

To obtain a dataset, you can download a public dataset from Hugging Face or ModelScope (choose one), such as the Alpaca dataset (in `.parquet` format):

```shell
mkdir -p ./dataset
cd ./dataset/
# Hugging Face dataset link
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
# ModelScope dataset link (recommended in China)
wget https://www.modelscope.cn/datasets/angelala00/tatsu-lab-alpaca/resolve/master/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```

> [!NOTE]
>
> - Local files in formats such as `.json`, `.jsonl`, `.csv`, `.parquet`, and `.txt` are automatically recognized by Hugging Face `datasets`. Datasets on the Hugging Face Hub (in `.parquet` and other formats) are also automatically recognized and loaded by Hugging Face `datasets`.
> - For downloading, format conversion, and validation of multimodal datasets, see [geo3k Dataset Fine-Tuning in Practice](./geo3k_finetune_guide.md).

## Starting Pretraining

In this stage, you will modify the pretraining configuration file and launch model pretraining. The steps are as follows:

1. Edit the training configuration file.

   ```shell
   vi examples/qwen3_5/config.yaml
   ```

2. Modify and save the training parameter configuration. An example configuration is as follows:

   ```yaml
   # --- Model configuration ---
   model:
     model_name_or_path: "./model_from_hf/qwen3_5_hf/"   # Model weight path
     tokenizer_name_or_path: "./model_from_hf/qwen3_5_hf/"  # Tokenizer path
     torch_dtype: "bf16"

   # --- Optimizer configuration ---
   optimizer:
     optimizer_type: "AdamW"
     lr: 3.0e-4
     weight_decay: 0.1
     adam_beta1: 0.9
     adam_beta2: 0.95
     warmup_ratio: 0.03
     lr_scheduler_type: "CosineAnnealingLR"
     min_lr: 3.0e-5
     clip_grad: 1.0

   # --- Run configuration ---
   run:
     seed: 42
     max_steps: 2000
     gradient_accumulation_steps: 1
     logging_steps: 1

   # --- Data configuration ---
   data:
     dataset_path: "./dataset/train.json"
     text_column: "text"
     split: "train"
     batch_size: 1
     max_seq_length: 4096
     num_workers: 8

   # --- Distributed configuration ---
   distributed:
     fully_shard_parallel_size: 8       # FSDP2 parallelism (recommended = number of devices per node)
     tensor_parallel_size: 1            # Tensor parallelism
     ulysses_parallel_size: 2           # Ulysses context parallelism
     fsdp_plan:
       param_dtype: "bf16"              # Computation precision
       reduce_dtype: "fp32"             # Gradient reduction precision
       num_to_forward_prefetch: 1       # Number of layers to prefetch in the forward pass
       num_to_backward_prefetch: 1      # Number of layers to prefetch in the backward pass
       hook_modules:
         - model.language_model.layers.{*}

   # --- Memory configuration ---
   memory:
     recompute: true
     recompute_plan:
       - model.language_model.layers.{*}
   ```

3. Run the pretraining script.

   ```shell
   cd examples/qwen3_5
   bash run.sh
   ```

   `run.sh` launches training via `torchrun`; cluster parameters can be overridden with environment variables:

   ```bash
   NNODES=1                 # Number of nodes, 1 for single-node
   GPUS_PER_NODE=16         # Number of GPUs per node
   MASTER_ADDR=127.0.0.1    # Master node IP
   MASTER_PORT=29500        # Master node port
   ```

   After running for a while, the terminal will display:

   ```shell
   Step 1 | Loss: 1.304194 | Grad Norm: 8.5150 | Step Time: 3.2145s | Memory: 28.3124 GB | Peak Memory: 46.8732 GB
   Step 2 | Loss: 1.232217 | Grad Norm: 4.3460 | Step Time: 2.8761s | Memory: 28.3056 GB | Peak Memory: 46.8732 GB
   Step 3 | Loss: 1.134654 | Grad Norm: 1.5500 | Step Time: 2.7934s | Memory: 28.2988 GB | Peak Memory: 46.8732 GB
   ```

   Once iterations begin, training is running normally.

> [!NOTE]
>
> - For multi-node training, start the pretraining script simultaneously in multiple terminals (each terminal has a different `NODE_RANK` and the same `MASTER_ADDR`/`MASTER_PORT`); torchrun handles the network setup.
> - The FSDP2 backend automatically shards model parameters across devices, ensuring that each device stores only a portion of the parameters, which enables training of ultra-large-scale models.
> - Training weights are saved to the directory specified by `checkpoint.output_dir`.

## Starting Fine-Tuning

In this stage, you will fine-tune the model using instruction data based on the pretrained weights. The steps are as follows:

1. Edit the training configuration file.

   ```shell
   vi examples/qwen3_5/config.yaml
   ```

2. Modify and save the fine-tuning parameter configuration. Compared with pretraining, the main changes are in the optimizer and data configurations:

   ```yaml
   # --- Optimizer configuration (smaller learning rate for fine-tuning) ---
   optimizer:
     optimizer_type: "AdamW"
     lr: 1.0e-5
     weight_decay: 0.01
     warmup_ratio: 0.0
     lr_scheduler_type: "cosine"
     min_lr: 1.0e-6
     clip_grad: 1.0

   # --- Data configuration (conversation-format data) ---
   data:
     dataset_path: "./dataset/finetune.json"
     conversation_column: "messages"
     split: "train"
     batch_size: 1
     max_seq_length: 4096
     num_workers: 8
   ```

3. Run the fine-tuning script.

   ```shell
   cd examples/qwen3_5
   bash run.sh
   ```

   After running for a while, the terminal will display:

   ```shell
   Step 1 | Loss: 3.316887 | Grad Norm: 70.9590 | Step Time: 3.0872s | Memory: 28.3124 GB | Peak Memory: 46.8732 GB
   Step 2 | Loss: 2.443476 | Grad Norm: 41.9860 | Step Time: 2.8145s | Memory: 28.3056 GB | Peak Memory: 46.8732 GB
   Step 3 | Loss: 2.669216 | Grad Norm: 45.3350 | Step Time: 2.7691s | Memory: 28.2988 GB | Peak Memory: 46.8732 GB
   ```

   Once iterations begin, training is running normally.

> [!NOTE]
>
> For multi-node fine-tuning, start the fine-tuning script simultaneously in multiple terminals (each terminal has a different `NODE_RANK`; all other parameters are the same).

## Training Script Parameters

The following table explains some key parameters in `config.yaml`:

**Table 1** Description of training configuration parameters

| Parameter | Description | Example Configuration |
|---|---|---|
| `model_name_or_path` | Hugging Face weight directory path; replace it with the actual model path | `./model_from_hf/qwen3_5_hf/` |
| `dataset_path` | Path to the training dataset; supports local files or a Hugging Face dataset name | `./dataset/train.json` |
| `text_column` / `conversation_column` | Text column name / conversation column name | `text` / `messages` |
| `max_seq_length` | Maximum training sequence length | `4096` |
| `fully_shard_parallel_size` | FSDP2 fully sharded data parallelism size | `8` |
| `tensor_parallel_size` | Tensor parallelism size | `1` |
| `ulysses_parallel_size` | Ulysses context parallelism size | `2` |
| `expert_parallel_size` | Expert parallelism size (set to `1` for Dense models) | `1` |
| `recompute` | Whether to enable activation recomputation, trading compute overhead for memory | `true` |
| `batch_size` | Training batch size per device | `1` |
| `gradient_accumulation_steps` | Number of gradient accumulation steps | `1` |
| `max_steps` | Maximum number of training steps | `2000` |

## MoE Model Training

Using Qwen3.5-35B-A3B (`examples/qwen3_5_moe`) as an example, add **expert parallelism (EP)** configuration on top of the Dense configuration:

```yaml
distributed:
  fully_shard_parallel_size: 8
  expert_parallel_size: 8             # Expert parallelism
  expert_fully_shard_parallel_size: 1 # Expert FSDP2 parallelism (EFSDP)
  ep_plan:
    apply_modules:                    # Modules to which EP is applied (wildcard)
      - model.language_model.layers.{*}.mlp.experts
    dispatcher: "fused"               # Dispatcher: eager / fused / mc2 / domino
    apply_efsdp_modules:              # Modules to which EFSDP is applied
      - model.language_model.layers.{*}.mlp.experts
```

For details on expert parallelism (dispatcher types, custom EP forward), see [Expert Parallelism](../features/expert_parallel.md).

The MoE example is launched in the same way as the Dense example:

```shell
cd examples/qwen3_5_moe
bash run.sh
```

## Follow-up Procedure

- See more supported models: [Supported Models](../models/supported_models.md)
- Integrate a custom model: [Model Adaptation Guide](../develop/model_adaptation.md)
- Multimodal data processing and fine-tuning: [geo3k Dataset Fine-Tuning in Practice](./geo3k_finetune_guide.md)
- Training performance analysis: [Performance Data Profiling](../tools/profiling.md)
