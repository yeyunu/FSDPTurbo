import os

import torch
import torch_npu
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import (
    FSDPTurboConfig,
    FSDPPlanConfig,
    EPPlanConfig,
    TPPlanConfig,
    DistributedConfig,
    MemoryConfig,
)
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.utils.log import set_log_level
from fsdp_turbo.utils.random import set_seed


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = 'Qwen/Qwen3-30B-A3B'
DATASET_PATH = 'wikitext/wikitext-2-raw-v1'
SEED = 42
LOG_LEVEL = 'INFO'

# Training hyperparameters
BATCH_SIZE = 1
MAX_LENGTH = 512
LR = 1e-4
NUM_STEPS = 3

# Parallel configuration
TENSOR_PARALLEL_SIZE = 2
EXPERT_PARALLEL_SIZE = 8
EXPERT_FULLY_SHARD_PARALLEL_SIZE = 1
FULLY_SHARD_PARALLEL_SIZE = 4


def initialization():
    """Initialize distributed environment, accelerator compatibility, seed, and logging."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)

    set_accelerator_compatible(torch.npu)
    set_seed(SEED)
    set_log_level(LOG_LEVEL)


def get_model_tokenizer():
    """Load the Qwen3 model and tokenizer from pretrained weights."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    return model, tokenizer


def get_dataloader(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    """Load and tokenize the wikitext dataset, returning a DataLoader."""
    dataset = load_dataset(DATASET_PATH, split='train')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length, padding='max_length')

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    def add_labels(examples):
        examples['labels'] = examples['input_ids'].clone()
        return examples

    tokenized_dataset = tokenized_dataset.map(add_labels)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# ============================================================
# Complete configurations
# ============================================================
def get_complete_custom_config():
    """
    Get complete FSDP configuration with all features enabled (Custom Implementation).
    
    Features:
    - FSDP2 with fully sharded data parallel
    - Expert Parallel with eager dispatcher
    - Tensor Parallel
    - Recompute for memory optimization
    - Mixed Precision (BF16 param/output, FP32 reduce)
    - Hook Module for fine-grained control
    - Prefetch for communication optimization
    """
    return FSDPTurboConfig(
        distributed=DistributedConfig(
            fully_shard_parallel_size=FULLY_SHARD_PARALLEL_SIZE,
            fsdp_plan=FSDPPlanConfig(
                ignored_modules=[],
                apply_modules={
                    'model.layers.{*}': {},
                    'model.embed_tokens': {},
                    'lm_head': {},
                },
                param_dtype='bf16',
                reduce_dtype='fp32',
                output_dtype='bf16',
                hook_modules=['model.layers.{*}'],
                fsdp_implementation='custom',
                num_to_forward_prefetch=1,
                num_to_backward_prefetch=1,
            ),
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            tp_plan=TPPlanConfig(
                colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj'],
                rowwise_parallel=['*.o_proj'],
            ),
            expert_parallel_size=EXPERT_PARALLEL_SIZE,
            expert_fully_shard_parallel_size=EXPERT_FULLY_SHARD_PARALLEL_SIZE,
            ep_plan=EPPlanConfig(
                apply_modules=['model.layers.{*}.mlp.experts'],
                dispatcher='fused',
            ),
        ),
        memory=MemoryConfig(
            recompute=True,
            recompute_plan=['model.layers.{*}'],
        ),
    )


# ============================================================
# Training loop
# ============================================================
def train_step(model, batch, optimizer):
    """Execute a single training step: forward, backward, optimizer step, zero_grad."""
    input_ids = batch['input_ids'].npu()
    attention_mask = batch['attention_mask'].npu()
    labels = batch['labels'].npu()

    outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    if torch.accelerator.current_device_index() == 0:
        print(f'[Rank {torch.accelerator.current_device_index()}] Loss: {loss.item()}')

    loss.backward()
    optimizer.step()
    model.zero_grad()

    return loss


def trainer_with_config(config, config_name, description, num_steps=NUM_STEPS, enable_profiler=False):
    """
    Train with a specific configuration.

    Args:
        config: FSDPTurboConfig instance.
        config_name: Name of the configuration.
        description: Description of the configuration.
        num_steps: Number of training steps to run.
        enable_profiler: Whether to enable NPU profiler.
    """
    initialization()

    if torch.accelerator.current_device_index() == 0:
        print(f'\n{"="*60}')
        print(f'Test: {config_name}')
        print(f'Description: {description}')
        print(f'{"="*60}\n')

    model, tokenizer = get_model_tokenizer()
    model = FSDPTurbo(config, model)
    dataloader = get_dataloader(tokenizer)

    if torch.accelerator.current_device_index() == 0:
        print(model)
    model.model.train(True)

    optimizer = AdamW(model.model.parameters(), lr=LR)

    if enable_profiler:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
        )
        prof_ctx = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=5000),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./profile'),
        )
    else:
        from contextlib import nullcontext
        prof_ctx = nullcontext()

    with prof_ctx as prof:
        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break

            torch.distributed.barrier()
            print(f'[Rank {torch.accelerator.current_device_index()}] start step {i}')

            loss = train_step(model, batch, optimizer)

            if enable_profiler:
                prof.step()

            torch.distributed.barrier()
            if torch.accelerator.current_device_index() == 0:
                print(
                    f'[Rank {torch.accelerator.current_device_index()}] step {i} '
                    f'memory: {torch.npu.memory_allocated() / 1024**3:.4f} GB '
                    f'max memory: {torch.npu.max_memory_allocated() / 1024**3:.4f} GB',
                )

    if torch.accelerator.current_device_index() == 0:
        print(f'\n Test completed: {config_name}\n')


# ============================================================
# Unified test entry point
# ============================================================
def test_qwen3():
    """
    Test Qwen3 with complete FSDP configurations.
    
    This test will run the following configuration.:
    All features enabled (FSDP2, EP, TP, Recompute, Mixed Precision, Hook Module, Prefetch)
    """

    trainer_with_config(
        config=get_complete_custom_config(),
        config_name='complete_custom',
        description='FSDP2 + EP + TP + Recompute + Mixed Precision + Hook Module + Prefetch (Custom)',
    )

    if torch.accelerator.current_device_index() == 0:
        print(f'\n{"#"*60}')
        print(f'All tests completed successfully!')
        print(f'{"#"*60}\n')


if __name__ == '__main__':
    test_qwen3()
