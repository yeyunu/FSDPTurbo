# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""Dataloader creation utilities.

Builds a ``DataLoader`` from a ``DataConfig``, optionally tokenizing
the dataset with a HuggingFace tokenizer/processor and wrapping it with a
``DistributedSampler`` for multi-process training.

Supports both pure-text and multimodal (text+image) datasets. Multimodal
records use configurable image and conversation columns and are converted
through the processor's chat template before collation.
"""

import json
import logging

import torch

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


def build_dataloader(data_cfg, tokenizer=None, processor=None):
    """Build a DataLoader from a HuggingFace dataset.

    Args:
        data_cfg: A ``DataConfig`` dataclass instance.
        tokenizer: An optional HuggingFace tokenizer. When provided,
            the dataset is tokenized and formatted for language modeling.
        processor: An optional multimodal processor. It formats conversations,
            loads image inputs, and creates batched model inputs.

    Returns:
        A ``torch.utils.data.DataLoader``, or ``None`` if
        ``data_cfg.dataset_path`` is empty.
    """
    if data_cfg is None or not data_cfg.dataset_path:
        print_rank(logger.warning, "dataset_path is empty, skipping dataloader build.")
        return None

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("`datasets` library is required for build_dataloader. Install it with: pip install datasets")

    print_rank(
        logger.info,
        f"> Building DataLoader for {data_cfg.dataset_path} {data_cfg.dataset_config} {data_cfg.split}",
    )
    if data_cfg.is_multimodal:
        if processor is None:
            raise ValueError("A processor is required when data.is_multimodal is enabled.")
        dataset = load_dataset("json", data_files=data_cfg.dataset_path, split=data_cfg.split)
        dataset = _process_multimodal_dataset(dataset, data_cfg, processor)
        collate_fn = _MultimodalCollator(processor, data_cfg.image_column)
    else:
        dataset = load_dataset(data_cfg.dataset_path, data_cfg.dataset_config, split=data_cfg.split)
        dataset = _process_text_dataset(dataset, data_cfg, tokenizer) if tokenizer is not None else dataset
        collate_fn = None

    sampler = None
    if torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        from fsdp_turbo.distributed.parallel_state import get_parallel_state

        parallel_state = get_parallel_state()
        sampler = DistributedSampler(
            dataset,
            num_replicas=parallel_state.get_data_group_size(),
            rank=parallel_state.get_data_rank(),
            shuffle=data_cfg.shuffle,
        )
        shuffle = False
    else:
        shuffle = data_cfg.shuffle

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
    )


def _process_text_dataset(dataset, data_cfg, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples[data_cfg.text_column],
            truncation=True,
            max_length=data_cfg.max_seq_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].clone()
        return examples

    return dataset.map(add_labels)


def _process_multimodal_dataset(dataset, data_cfg, processor):
    conversation_column = data_cfg.conversation_column
    image_column = data_cfg.image_column

    def process_example(example):
        messages_raw = example[conversation_column]
        if isinstance(messages_raw, str):
            messages_raw = json.loads(messages_raw)

        images_raw = example.get(image_column, None)
        if isinstance(images_raw, str):
            images_raw = json.loads(images_raw)

        image_paths = images_raw or []
        messages = []
        image_inserted = False
        for message in messages_raw:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                if image_paths and not image_inserted:
                    user_content = [{"type": "image"}]
                    text = content.replace("<image>", "").strip()
                    if text:
                        user_content.append({"type": "text", "text": text})
                    messages.append({"role": "user", "content": user_content})
                    image_inserted = True
                else:
                    messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})

        return {
            "text": processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
            image_column: image_paths,
        }

    return dataset.map(process_example, remove_columns=dataset.column_names)


class _MultimodalCollator:
    def __init__(self, processor, image_column):
        self.processor = processor
        self.image_column = image_column

    def __call__(self, features):
        from PIL import Image

        texts = [feature["text"] for feature in features]
        images = []
        for feature in features:
            for path in feature.get(self.image_column, []):
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))

        processor_kwargs = {
            "text": texts,
            "padding": "longest",
            "truncation": True,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images

        batch = self.processor(**processor_kwargs)
        labels = batch["input_ids"].clone()
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.processor.tokenizer, "image_token_id", None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch
