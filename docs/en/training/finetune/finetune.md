# Model Fine-Tuning

## Application Scenarios

Supervised fine-tuning (SFT) continues training a pretrained model with high-quality instruction and response data, enabling the model to learn specific tasks, domain knowledge, or conversational styles. This method is suitable for scenarios such as question answering, text generation, summarization, translation, code generation, and domain adaptation.

> [!NOTE]
>
> If you are using FSDPTurbo for the first time, we recommend first completing an end-to-end walkthrough with the [Quick Start](../quick_start_fsdpturbo.md) (Qwen3.5-27B pretraining and fine-tuning). This document focuses on the model, dataset, YAML configuration, and parameters for full-parameter fine-tuning, and is suitable for scenarios where you switch models or use custom datasets.

This document describes how to perform full-parameter fine-tuning with FSDPTurbo based on a Hugging Face-format pretrained model. The example uses the Qwen3.5-27B model. The main workflow is as follows:

1. Set up the environment.
2. Prepare model weights.
3. Prepare the dataset.
4. Configure fine-tuning parameters.
5. Configure the launch script.
6. Start fine-tuning.

## Environment Setup

Before starting fine-tuning, complete the environment installation by following [Software Installation](../install_guide.md).

## Model Weight Preparation

For the model weight download URLs, see the [Supported Models list](../../models/supported_models.md). This example uses the Hugging Face-format weights of Qwen3.5-27B.

FSDPTurbo can directly load Hugging Face-format weights — **no model weight conversion is required before training**. After loading, the training script converts the dtype of model parameters and buffers according to the `torch_dtype` configuration. Simply set the model weight directory in `model.model_name_or_path` in `config.yaml`:

```yaml
model:
  model_name_or_path: "./model_from_hf/qwen3_5_hf/"
  tokenizer_name_or_path: "./model_from_hf/qwen3_5_hf/"
  torch_dtype: "bf16"
```

## Dataset Preparation

FSDPTurbo data loading is implemented based on Hugging Face `datasets` and supports local data files or Hugging Face Hub datasets. Loading, format alignment, and tokenization are completed when the training task starts; no prior conversion to Megatron Indexed Dataset is required.

### Supported Data Formats

- **Plain text format**: Each sample contains a `text` field, used for pretraining or pre-concatenated instruction fine-tuning data.

    ```json
    [
      {"text": "User: Please introduce FSDPTurbo\nAssistant: FSDPTurbo is a distributed training acceleration library..."}
    ]
    ```

- **Conversation format**: Specify the conversation column name with `conversation_column`; each message contains `role` and `content` fields, suitable for instruction fine-tuning and multimodal fine-tuning.

    ```json
    [
      {
        "messages": [
          {"role": "user", "content": "Please introduce FSDPTurbo"},
          {"role": "assistant", "content": "FSDPTurbo is a distributed training acceleration library..."}
        ]
      }
    ]
    ```

    For multimodal scenarios, add an `images` field to the sample and use the `<image>` placeholder in `content`:

    ```json
    [
      {
        "messages": [
          {"role": "user", "content": "<image>\nPlease describe this image"},
          {"role": "assistant", "content": "This image shows..."}
        ],
        "images": ["/path/to/image1.jpg"]
      }
    ]
    ```

> [!NOTE]
>
> Local files in formats such as `.json`, `.jsonl`, `.csv`, `.parquet`, and `.txt` are automatically recognized by Hugging Face `datasets`. Multimodal data currently supports JSON format files only.

## Configuring Fine-Tuning Parameters

Parameters such as the model weight path, dataset path, parallelism scale, and output directory are configured in `config.yaml`. For detailed configuration, see the example `examples/qwen3_5/config.yaml`. An example YAML is as follows:

```yaml
model:
  model_name_or_path: "./model_from_hf/qwen3_5_hf/"
  tokenizer_name_or_path: "./model_from_hf/qwen3_5_hf/"
  torch_dtype: "bf16"

optimizer:
  optimizer_type: "AdamW"
  lr: 1.0e-5
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_epsilon: 1e-8
  clip_grad: 1.0
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.0
  min_lr: 1.0e-6

run:
  seed: 42
  max_steps: 2000
  gradient_accumulation_steps: 1
  logging_steps: 1

checkpoint:
  output_dir: "./output/qwen3_5_27b"
  save_steps: 500

data:
  dataset_path: "./dataset/finetune.json"
  conversation_column: "messages"
  split: "train"
  batch_size: 1
  max_seq_length: 4096
  num_workers: 8

distributed:
  fully_shard_parallel_size: 8
  tensor_parallel_size: 1
  ulysses_parallel_size: 2
  fsdp_plan:
    param_dtype: "bf16"
    reduce_dtype: "fp32"
    num_to_forward_prefetch: 1
    num_to_backward_prefetch: 1
    hook_modules:
      - model.language_model.layers.{*}

memory:
  recompute: true
  recompute_plan:
    - model.language_model.layers.{*}
```

The main parameters in the YAML are described as follows:

| Parameter | Description | Default Value |
|---|---|---|
| `model_name_or_path` | Hugging Face model weight directory path. | — |
| `dataset_path` | Training dataset path; supports local files or a Hugging Face dataset name. | — |
| `conversation_column` | Column name of the conversation-format data. | `messages` |
| `max_seq_length` | Maximum length of the tokenized training sequence; content beyond this length is truncated. | `4096` |
| `fully_shard_parallel_size` | FSDP2 fully sharded data parallelism size; should match the number of devices per node. | `1` |
| `recompute` | Whether to enable activation recomputation, trading compute overhead for memory. | `false` |
| `lr` | Initial learning rate. | `1e-4` |
| `max_steps` | Maximum number of training steps; overrides `num_train_epochs` when greater than 0. | `-1` |
| `save_steps` | Step interval for saving training checkpoints. | `500` |

## Configuring the Launch Script

Open the corresponding `run.sh` file and configure the distributed training parameters. For detailed configuration, see the example `examples/qwen3_5/run.sh`.

A single-node 16-device configuration is as follows:

```bash
NNODES=1                              # Total number of nodes, set to 1 for single-node
GPUS_PER_NODE=16                      # Number of GPUs per node
MASTER_ADDR=127.0.0.1                 # Master node IP, can be set to 127.0.0.1 for single-node
MASTER_PORT=29500                     # Master node communication port
```

An example multi-node configuration is as follows:

```bash
NNODES=2                              # Total number of nodes participating in training
GPUS_PER_NODE=16                      # Number of GPUs per node
MASTER_ADDR="<master node IP>"        # All nodes are configured with the master node IP; localhost cannot be used
MASTER_PORT=29500                     # All nodes use the same master node communication port
```

> [!NOTE]
>
> `MASTER_ADDR`, `MASTER_PORT`, and `NNODES` must be consistent across nodes, and `NODE_RANK` (controlled by torchrun `--nnodes` and `--node_rank`) starts from 0 and must not be duplicated.

`run.sh` launches training via `torchrun`:

```bash
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config config.yaml
```

## Starting Fine-Tuning

After the parameter configuration is complete, run the following in the example directory:

```shell
cd examples/qwen3_5
bash run.sh
```

For multi-node training, run the launch script on all nodes and set the corresponding `NODE_RANK` on each. Training checkpoints are saved to the directory specified by `checkpoint.output_dir`.

After running for a while, you can see logs like the following in the terminal:

```shell
Step 1 | Loss: 3.316887 | Grad Norm: 70.9590 | Step Time: 3.0872s | Memory: 28.3124 GB | Peak Memory: 46.8732 GB
Step 2 | Loss: 2.443476 | Grad Norm: 41.9860 | Step Time: 2.8145s | Memory: 28.3056 GB | Peak Memory: 46.8732 GB
Step 3 | Loss: 2.669216 | Grad Norm: 45.3350 | Step Time: 2.7691s | Memory: 28.2988 GB | Peak Memory: 46.8732 GB
```

Once the terminal continuously outputs information such as the step number, loss value, gradient norm, per-step time, and memory usage, the training task is running normally.

## Usage Constraints

- This guide applies to FSDPTurbo full-parameter SFT and does not cover other post-training methods such as LoRA or DPO.
- `model_name_or_path`, the dataset, and the output directory should use valid paths accessible in the training environment.
- The model size, sequence length, batch size, and parallelism scale need to be adjusted according to the number of devices and the memory capacity.
- For multi-node training, ensure that all nodes can correctly access the model and dataset paths.
