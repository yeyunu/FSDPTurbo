"""DeepSeekV4 CI baseline system test.

Runs a single merged config (all compatible features enabled) for
NUM_STEPS (default 10) and prints per-step metrics in the standard
format consumed by acquire_json.py.
"""

import os
import time

import torch
import yaml

try:
    import torch_npu
except ImportError:
    torch_npu = None

from fsdp_turbo.fsdp_turbo import FSDPTurbo
from fsdp_turbo.fsdp_turbo_config import load_config_from_yaml
from fsdp_turbo.quantization.cache import clear_weight_cache
from fsdp_turbo.training.clip_grads import compute_grad_norm
from fsdp_turbo.utils.device import set_accelerator_compatible
from fsdp_turbo.utils.log import set_log_level
from fsdp_turbo.utils.random import set_seed


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
SEED = 42
LOG_LEVEL = "INFO"
NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))


def get_model(config, features):
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Config
    import modeling_deepseek_v4  # noqa: F401

    model_config = DeepseekV4Config()
    model_config.num_hidden_layers = 4
    model_config.n_routed_experts = 16
    model_config.index_topk = 16

    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(model_config)

    for param in model.parameters():
        param.data = param.data.to(config.model.torch_dtype)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(config.model.torch_dtype)

    init_weights = model._init_weights

    model = FSDPTurbo(config, model)
    model.to_empty(device="npu")

    with torch.no_grad():
        model.apply(init_weights)

    if features.get("swap_activation"):
        from fsdp_turbo.memory.swap_activation.swap_activation import (
            swap_activation_modules,
        )

        plan = features.get("swap_activation_plan") or ["model.layers.{*}"]
        model = swap_activation_modules(model, plan)

    for name, module in model.named_modules():
        if "indexer" in name:
            for n, p in module.named_parameters():
                p.grad = torch.zeros_like(p.data)

    return model


def get_dataloader(config):
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    data_cfg = config.data
    dataset = load_from_disk(data_cfg.dataset_path)
    if isinstance(dataset, dict):
        dataset = dataset[data_cfg.split]

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item.get("attention_mask", []) for item in batch]
        labels = [item.get("labels", []) for item in batch]
        max_len = max(len(seq) for seq in input_ids)

        padded_input_ids, padded_attention_mask, padded_labels = [], [], []
        for i in range(len(batch)):
            seq_len = len(input_ids[i])
            pad_len = max_len - seq_len
            padded_input_ids.append(input_ids[i] + [pad_token_id] * pad_len)
            if attention_mask[i]:
                padded_attention_mask.append(attention_mask[i] + [0] * pad_len)
            else:
                padded_attention_mask.append([1] * seq_len + [0] * pad_len)
            if labels[i]:
                padded_labels.append(labels[i] + [-100] * pad_len)
            else:
                padded_labels.append([-100] * max_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=data_cfg.shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
    )


def get_optimizer(model, config, features):
    from fsdp_turbo.quantization.cache import hook_optimizer_step

    optim_cfg = config.optimizer
    if features.get("swap_optimizer"):
        from fsdp_turbo.memory.swap_optimizer.swap_optimizer import AdamWSwap

        optimizer = AdamWSwap(
            model.parameters(),
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.weight_decay,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            eps=optim_cfg.adam_epsilon,
        )
    else:
        from torch.optim import AdamW

        optimizer = AdamW(
            model.parameters(),
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.weight_decay,
            betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
            eps=optim_cfg.adam_epsilon,
        )
    hook_optimizer_step(model, optimizer)
    return optimizer


def train_step(model, batch, optimizer, features, gas):
    batch["use_cache"] = False

    if features.get("chunk_loss"):
        loss = _compute_chunk_loss(model, batch, features.get("chunk_loss_size", 1024))
    else:
        outputs = model(**batch)
        loss = outputs.loss

    loss = loss / gas
    loss.backward()
    grad_norm = compute_grad_norm(list(model.parameters()))
    optimizer.step()
    clear_weight_cache()
    model.zero_grad()
    return loss, grad_norm


def _compute_chunk_loss(model, batch, chunk_size):
    from fsdp_turbo.memory.chunk_loss.chunk_loss import calculate_lm_loss, chunk_loss

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask", None),
        use_cache=False,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]
    hidden = hidden_states[:, :-1, :].contiguous()

    head = model.model.lm_head
    shift_labels = batch["labels"][:, 1:].contiguous().to(hidden.device)
    num_items = (shift_labels != -100).sum().clamp(min=1)

    loss_kwargs_chunks = [
        {
            "shift_labels": chunk,
            "alpha": num_items,
            "ignore_index": -100,
            "reduction": "sum",
        }
        for chunk in torch.split(shift_labels, chunk_size, dim=1)
    ]
    return chunk_loss(
        hidden,
        head.weight,
        getattr(head, "bias", None),
        calculate_lm_loss,
        loss_kwargs_chunks,
        chunk_size,
    )


def test_deepseekv4():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)
    set_accelerator_compatible()
    set_seed(SEED)
    set_log_level(LOG_LEVEL)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = load_config_from_yaml(CONFIG_PATH)
    features = raw.get("features") or {}
    gas = config.run.gradient_accumulation_steps

    if torch.accelerator.current_device_index() == 0:
        print(f"DeepSeekV4 CI baseline: features={[k for k, v in features.items() if v]}")

    model = get_model(config, features)
    dataloader = get_dataloader(config)
    optimizer = get_optimizer(model, config, features)
    model.train(True)

    for step, batch in enumerate(dataloader):
        if step >= NUM_STEPS:
            break

        torch.distributed.barrier()
        torch.npu.reset_peak_memory_stats()

        step_start = time.perf_counter()
        loss, grad_norm = train_step(model, batch, optimizer, features, gas)
        torch.npu.synchronize()
        step_time = time.perf_counter() - step_start

        torch.distributed.barrier()
        if torch.accelerator.current_device_index() == 0:
            print(
                f"[Rank {torch.accelerator.current_device_index()}] step {step} "
                f"step time: {step_time:.4f}s "
                f"loss: {loss.item():.4f} "
                f"grad norm: {grad_norm:.4f} "
                f"memory: {torch.npu.memory_allocated() / 1024**3:.4f} GB "
                f"max memory: {torch.npu.max_memory_allocated() / 1024**3:.4f} GB",
            )

    if torch.accelerator.current_device_index() == 0:
        print("DeepSeekV4 CI baseline: completed.")


if __name__ == "__main__":
    test_deepseekv4()
