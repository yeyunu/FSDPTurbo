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
import math
import os
from dataclasses import replace

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
        if _is_local_jsonl(data_cfg.dataset_path):
            dataset = _LocalJsonlMultimodalDataset(data_cfg.dataset_path, data_cfg, processor)
        else:
            dataset = load_dataset("json", data_files=data_cfg.dataset_path, split=data_cfg.split)
            dataset = _process_multimodal_dataset(dataset, data_cfg, processor)
        collate_fn = _MultimodalCollator(
            processor,
            data_cfg.image_column,
            data_cfg.media_key,
            data_cfg=data_cfg,
        )
    else:
        if os.path.isfile(data_cfg.dataset_path):
            # datasets >= 5 rejects a bare local file path as a builder name
            # ("Couldn't find any data file at ..."); route it through the json
            # loader exactly like the multimodal branch.
            dataset = load_dataset("json", data_files=data_cfg.dataset_path, split=data_cfg.split)
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


def _is_local_jsonl(dataset_path):
    return (
        isinstance(dataset_path, (str, os.PathLike))
        and os.path.isfile(dataset_path)
        and str(dataset_path).lower().endswith((".jsonl", ".jsonlines"))
    )


class _LocalJsonlMultimodalDataset(torch.utils.data.Dataset):
    """Lazily read local JSONL without building a Hugging Face Arrow cache."""

    def __init__(self, path, data_cfg, processor):
        self.path = os.fspath(path)
        self.data_cfg = data_cfg
        self.processor = processor
        self.offsets = []
        with open(self.path, "rb") as stream:
            while stream.readline():
                self.offsets.append(stream.tell())
        self.offsets.insert(0, 0)
        self.offsets.pop()

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with open(self.path, "rb") as stream:
            stream.seek(self.offsets[index])
            example = json.loads(stream.readline())
        return _process_multimodal_example(example, self.data_cfg, self.processor)


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
    def process_example(example):
        return _process_multimodal_example(example, data_cfg, processor)

    return dataset.map(process_example, remove_columns=dataset.column_names)


def _process_multimodal_example(example, data_cfg, processor):
    conversation_column = data_cfg.conversation_column
    image_column = data_cfg.image_column
    if example.get("documents"):
        documents = []
        for document in example["documents"]:
            rendered = _render_multimodal_record(
                document.get(conversation_column, []),
                document.get(image_column, []),
                processor,
                image_column,
            )
            documents.append(rendered)
        return {
            "documents": documents,
            "sfa_layout": data_cfg.sfa_layout or "TND",
        }

    if data_cfg.sfa_layout == "TND":
        raise ValueError("data.sfa_layout=TND requires packed records with documents.")

    return _render_multimodal_record(
        example[conversation_column],
        example.get(image_column, None),
        processor,
        image_column,
    )


def _build_messages_with_image_markers(messages_raw, image_paths):
    """Preserve image-marker positions and validate their correspondence to images."""
    marker_count = sum(
        message.get("content", "").count("<image>") for message in messages_raw if message.get("role") == "user"
    )
    if marker_count and marker_count != len(image_paths):
        raise ValueError(f"Found {marker_count} <image> markers but received {len(image_paths)} images.")

    messages = []
    inserted_images = 0
    for message in messages_raw:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "user":
            if marker_count and "<image>" in content:
                parts = content.split("<image>")
                user_content = []
                for index, part in enumerate(parts):
                    if part:
                        user_content.append({"type": "text", "text": part})
                    if index < len(parts) - 1:
                        user_content.append({"type": "image"})
                        inserted_images += 1
                messages.append({"role": "user", "content": user_content})
            elif not marker_count and image_paths and not inserted_images:
                user_content = [{"type": "image"} for _ in image_paths]
                if content:
                    user_content.append({"type": "text", "text": content})
                messages.append({"role": "user", "content": user_content})
                inserted_images = len(image_paths)
            else:
                messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})

    if inserted_images != len(image_paths):
        raise ValueError(f"Rendered {inserted_images} image markers for {len(image_paths)} images.")
    return messages


def _render_multimodal_record(messages_raw, images_raw, processor, image_column):
    if isinstance(messages_raw, str):
        messages_raw = json.loads(messages_raw)

    if isinstance(images_raw, str):
        images_raw = json.loads(images_raw)

    image_paths = images_raw or []
    if getattr(processor, "preserves_image_marker_positions", False):
        messages = _build_messages_with_image_markers(messages_raw, image_paths)
    else:
        messages = []
        image_inserted = False
        for message in messages_raw:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                if image_paths and not image_inserted:
                    user_content = [{"type": "image"} for _ in image_paths]
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


class _MultimodalCollator:
    def __init__(self, processor, image_column, media_key="images", data_cfg=None, max_seq_length=None):
        self.processor = processor
        self.image_column = image_column
        self.media_key = media_key
        self.data_cfg = data_cfg
        self.max_seq_length = (
            max_seq_length if max_seq_length is not None else getattr(data_cfg, "max_seq_length", None)
        )

    def __call__(self, features):
        from PIL import Image

        if any(feature.get("documents") for feature in features):
            if len(features) != 1:
                raise ValueError("Packed multimodal records require data.batch_size=1.")
            layout = features[0].get("sfa_layout", "TND")
            if layout == "TND":
                return self._collate_tnd(features[0]["documents"], Image)
            if layout == "BSND":
                return self._collate_bsnd_documents(features[0]["documents"], Image)
            raise ValueError(f"Unsupported packed sfa_layout: {layout!r}")

        texts = [feature["text"] for feature in features]
        images = []
        for feature in features:
            for path in feature.get(self.image_column, []):
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))

        # When a batch has no images, create a dummy image and append fake image
        # tokens so that every rank calls the vision encoder forward. Without this,
        # ranks with text-only samples skip self.visual(...) entirely, causing the
        # FSDP all-gather for the vision encoder to hang (HCCL timeout). The fake
        # span is delimited by vision_start_token_id; processors that do not expose
        # it cannot mask the span afterwards, so they keep the pre-existing
        # behavior of not appending anything.
        vision_start_token_id = getattr(self.processor.tokenizer, "vision_start_token_id", None)
        fake_image_appended = False
        if (
            not images
            and vision_start_token_id is not None
            and not getattr(self.processor, "supports_empty_image_batch", False)
        ):
            dummy_image = Image.new("RGB", (64, 64), (255, 255, 255))
            images = [dummy_image]
            fake_messages = [{"role": "user", "content": [{"type": "image"}]}]
            fake_text = self.processor.apply_chat_template(fake_messages, tokenize=False, add_generation_prompt=False)
            texts[0] = texts[0] + fake_text
            fake_image_appended = True

        processor_kwargs = {
            "text": texts,
            "padding": "longest",
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.max_seq_length is not None and getattr(self.processor, "supports_image_safe_truncation", False):
            processor_kwargs["max_length"] = self.max_seq_length
        if images:
            if self.media_key == "medias":
                medias = [
                    image
                    if isinstance(image, dict) and image.get("type") == "image"
                    else {"type": "image", "image": image}
                    for image in images
                ]
                processor_kwargs["medias"] = medias
            else:
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
        media_token_id = getattr(self.processor.tokenizer, "media_placehodler_token_id", None)
        if media_token_id is not None:
            labels[labels == media_token_id] = -100

        # Mask the fake image tokens appended to the first sample so they do not
        # affect attention or loss. The original text had no vision tokens, so the
        # first vision_start_token_id marks the beginning of the fake text.
        if fake_image_appended:
            vs_positions = (batch["input_ids"][0] == vision_start_token_id).nonzero(as_tuple=True)[0]
            if len(vs_positions) > 0:
                fake_start = vs_positions[0].item()
                batch["attention_mask"][0, fake_start:] = 0
                labels[0, fake_start:] = -100

        batch["labels"] = labels
        return batch

    def _process_packed_documents(self, documents, image_module):
        if not documents:
            raise ValueError("Packed training requires at least one document.")
        max_seq_length = getattr(
            self.data_cfg,
            "max_seq_length",
            getattr(getattr(self.processor, "model_args", None), "max_seq_len", 4096),
        )
        parts = []
        compression_alignment = math.lcm(
            *(
                ratio
                for ratio in getattr(
                    getattr(self.processor, "model_args", None),
                    "compress_ratios",
                    (1,),
                )
                if ratio > 0
            )
        )
        for document in documents:
            images = []
            for path in document.get(self.image_column, []):
                with image_module.open(path) as image:
                    images.append(image.convert("RGB"))
            part = self.processor(
                text=[document["text"]],
                images=images,
                padding="longest",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            pad_length = (-part["input_ids"].shape[1]) % compression_alignment
            if pad_length:
                pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", 0)
                part["input_ids"] = torch.nn.functional.pad(part["input_ids"], (0, pad_length), value=pad_token_id or 0)
                part["attention_mask"] = torch.nn.functional.pad(part["attention_mask"], (0, pad_length), value=0)
                part["token_types"] = torch.nn.functional.pad(part["token_types"], (0, pad_length), value=-1)
            parts.append(part)

        lengths = [part["input_ids"].shape[1] for part in parts]
        packed_total_tokens = sum(lengths)
        if packed_total_tokens > max_seq_length:
            raise ValueError(
                "Packed record exceeds data.max_seq_length after multimodal "
                f"expansion/alignment: document_lengths={lengths}, "
                f"packed_total_tokens={packed_total_tokens}, "
                f"max_seq_length={max_seq_length}. Repack the source data."
            )
        return parts, lengths, packed_total_tokens

    def _collate_bsnd_documents(self, documents, image_module):
        parts, lengths, _ = self._process_packed_documents(documents, image_module)
        max_batch_size = getattr(getattr(self.processor, "model_args", None), "max_batch_size", None)
        if max_batch_size is not None and len(parts) > max_batch_size:
            raise ValueError(f"BSND pack has {len(parts)} documents, but model.max_batch_size={max_batch_size}.")
        padded_length = max(lengths)
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("BSND packing requires a pad or EOS token ID.")

        def stack_padded(name, value):
            return torch.cat(
                [
                    torch.nn.functional.pad(part[name], (0, padded_length - length), value=value)
                    for part, length in zip(parts, lengths)
                ],
                dim=0,
            )

        input_ids = stack_padded("input_ids", pad_token_id)
        attention_mask = stack_padded("attention_mask", 0)
        token_types = stack_padded("token_types", -1)
        labels = input_ids.clone()
        labels[token_types >= 0] = -100
        labels[~attention_mask.bool()] = -100
        labels[labels == pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_types": token_types,
            "images": [part.get("images", [None])[0] for part in parts],
            "labels": labels,
            "sfa_layout": "BSND",
        }

    def _collate_tnd(self, documents, image_module):
        parts, lengths, _ = self._process_packed_documents(documents, image_module)
        actual_seq_len = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
        input_ids = torch.cat([part["input_ids"] for part in parts], dim=1)
        attention_mask = torch.cat([part["attention_mask"] for part in parts], dim=1)
        token_types = torch.cat([part["token_types"] for part in parts], dim=1)

        packed_images = []
        offset = 0
        for part, length in zip(parts, lengths):
            for image in (part.get("images") or [None])[0] or ():
                packed_images.append(replace(image, start=image.start + offset))
            offset += length

        labels = input_ids.clone()
        labels[token_types >= 0] = -100
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        position_ids = torch.cat([torch.arange(length, dtype=torch.long) for length in lengths]).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_types": token_types,
            "images": [packed_images or None],
            "labels": labels,
            "position_ids": position_ids,
            "sfa_layout": "TND",
            "actual_seq_len": actual_seq_len,
        }
