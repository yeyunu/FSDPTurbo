# Distributed Pretraining of Large Models

## Application Scenarios

Pretraining is a core step in the development of large language models. Its goal is to enable the model to learn language patterns and world knowledge from large-scale unlabeled corpora. The pretraining process focuses more on language modeling itself than on performing specific tasks. Taking GPT-style models as an example, they are typical autoregressive language models whose core idea is to predict the next token based on historical context. Through the pretraining process of repeatedly optimizing this predictive ability, the model gradually learns how to understand context, maintain sentence coherence, and master higher-level language structures, providing general language representation capabilities for a variety of downstream tasks.

Pretraining data is usually in plain text format and is not task-oriented. For example:

```json
{"text": "It is a nice day today; let us go hiking together."}
{"text": "Deep learning is changing the world."}
{"text": "The emergence of AI has driven the development of human society."}
```

> [!NOTE]
>
> - FSDPTurbo can directly load Hugging Face-format weights without conversion to Megatron or other formats (the dtype conversion after loading is done by the training script according to the `torch_dtype` configuration). If no initial weights are loaded, the model weights are randomly initialized and training starts from scratch.
> - Pretraining and fine-tuning use the same training entry point (`train.py`) and are distinguished by the data format, optimizer parameters, and so on in the YAML configuration.

## Instructions

Using the Qwen3.5-27B model as an example, this document describes how to start pretraining. Distributed pretraining of large models mainly involves the following steps:

1. Set up the environment.
2. Prepare model weights.
3. Prepare pretraining data.
4. Configure pretraining parameters.
5. Start pretraining.

### Environment Setup

Before starting pretraining, complete the environment installation by following [Software Installation](../install_guide.md), and make sure the relevant environment variables have been configured:

```shell
source /usr/local/Ascend/cann/set_env.sh
```

The command above uses the default path for a root user installation as an example; replace it with the actual path of `set_env.sh`.

### Model Weight Preparation

FSDPTurbo directly loads Hugging Face-format weights without conversion to Megatron or other formats. After loading, the training script converts the dtype of model parameters and buffers according to the `torch_dtype` configuration. To obtain the Qwen3.5-27B weights:

```shell
mkdir -p ./model_from_hf/qwen3_5_hf
huggingface-cli download Qwen/Qwen3.5-27B --local-dir ./model_from_hf/qwen3_5_hf
```

### Pretraining Data Preparation

FSDPTurbo data loading is implemented based on Hugging Face `datasets` — **no prior data preprocessing is required**; loading and tokenization are completed automatically when training starts. Local data files (`.json`, `.jsonl`, `.csv`, `.parquet`, `.txt`, and so on) and Hugging Face Hub datasets are supported.

Common pretraining datasets include:

- [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [EnWiki dataset](https://huggingface.co/datasets/lsb/enwiki20230101)
- [C4 dataset](https://huggingface.co/datasets/allenai/c4)
- [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)

Pretraining data must contain a `text` column, in the following format:

```json
[
    {"text": "document 1"},
    {"text": "document 2"}
]
```

Configure the dataset in the `data` section of `config.yaml`:

```yaml
data:
  dataset_path: "./dataset/train.json"    # Local file path or Hugging Face dataset name
  text_column: "text"                      # Text column name
  split: "train"
  batch_size: 1
  max_seq_length: 4096
  num_workers: 8
```

### Configuring Pretraining Parameters

Parameters such as the model weight path, dataset path, parallelism scale, and output directory are configured in `config.yaml`. For detailed configuration, see the example `examples/qwen3_5/config.yaml`. Key points of a typical pretraining configuration:

- A larger optimizer learning rate (for example, `3.0e-4`)
- Data uses the plain text format (`text_column`)
- Enable activation recomputation to reduce memory usage

```yaml
model:
  model_name_or_path: "./model_from_hf/qwen3_5_hf/"
  tokenizer_name_or_path: "./model_from_hf/qwen3_5_hf/"
  torch_dtype: "bf16"

optimizer:
  optimizer_type: "AdamW"
  lr: 3.0e-4
  weight_decay: 0.1
  warmup_ratio: 0.03
  lr_scheduler_type: "CosineAnnealingLR"
  min_lr: 3.0e-5
  clip_grad: 1.0

run:
  seed: 42
  max_steps: 2000
  gradient_accumulation_steps: 1
  logging_steps: 1

data:
  dataset_path: "./dataset/train.json"
  text_column: "text"
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

### Starting Pretraining

After the parameter configuration is complete, run the following in the example directory:

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

After running for a while, you can see logs like the following in the terminal:

```shell
Step 1 | Loss: 1.304194 | Grad Norm: 8.5150 | Step Time: 3.2145s | Memory: 28.3124 GB | Peak Memory: 46.8732 GB
Step 2 | Loss: 1.232217 | Grad Norm: 4.3460 | Step Time: 2.8761s | Memory: 28.3056 GB | Peak Memory: 46.8732 GB
Step 3 | Loss: 1.134654 | Grad Norm: 1.5500 | Step Time: 2.7934s | Memory: 28.2988 GB | Peak Memory: 46.8732 GB
```

Once the terminal continuously outputs information such as the step number, loss value, gradient norm, per-step time, and memory usage, the training task is running normally.

> [!NOTE]
>
> - For multi-node training, start the pretraining script simultaneously in multiple terminals (each terminal has a different `NODE_RANK` and the same `MASTER_ADDR`/`MASTER_PORT`); torchrun handles the network setup.
> - For multi-node training, ensure that all nodes can correctly access the model and dataset paths.
> - Training weights are saved to the directory specified by `checkpoint.output_dir`.
