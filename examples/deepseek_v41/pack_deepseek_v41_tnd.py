#!/usr/bin/env python3
"""Greedily pack normalized multimodal JSONL using real DeepSeek-V4.1 lengths."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image
from transformers import PreTrainedTokenizerFast

from fsdp_turbo.models.deepseek_v41 import (
    DeepSeekV41Processor,
    build_deepseek_v41_model_args,
    prepare_deepseek_v41_tokenizer,
)
from fsdp_turbo.training.dataloader import _render_multimodal_record


def iter_documents(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            documents = record.get("documents") or [record]
            for document in documents:
                yield line_number, document


def expanded_aligned_length(document, processor, alignment, max_seq_length):
    rendered = _render_multimodal_record(
        document.get("messages", []),
        document.get("images", []),
        processor,
        "images",
    )
    images = []
    try:
        for path in rendered["images"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        processed = processor(
            text=[rendered["text"]],
            images=images,
            padding="longest",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
    finally:
        for image in images:
            image.close()
    length = processed["input_ids"].shape[1]
    return math.ceil(length / alignment) * alignment


def write_pack(stream, documents):
    stream.write(
        json.dumps(
            {"documents": documents, "sfa_layout": "TND"},
            ensure_ascii=False,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description=("Pack normalized multimodal JSONL by the post-processor token length.")
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-documents-per-pack", type=int, default=0)
    parser.add_argument(
        "--oversized-document-policy",
        choices=("error", "skip"),
        default="error",
        help=(
            "How to handle a single document that exceeds max_seq_length after "
            "multimodal expansion. 'skip' is safe; image spans are never truncated."
        ),
    )
    args = parser.parse_args()

    if args.max_seq_length <= 0:
        parser.error("--max-seq-length must be positive")
    if args.max_documents_per_pack < 0:
        parser.error("--max-documents-per-pack cannot be negative")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    model_args = build_deepseek_v41_model_args()
    prepare_deepseek_v41_tokenizer(tokenizer, model_args)
    processor = DeepSeekV41Processor(tokenizer, model_args)
    alignment = math.lcm(*(ratio for ratio in model_args.compress_ratios if ratio > 0))

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_documents = []
    current_tokens = 0
    document_count = 0
    pack_count = 0
    multi_document_pack_count = 0
    max_packed_tokens = 0
    skipped_oversized_documents = 0

    with output_path.open("w", encoding="utf-8") as output_stream:
        for line_number, document in iter_documents(input_path):
            try:
                length = expanded_aligned_length(document, processor, alignment, args.max_seq_length)
            except ValueError as error:
                if args.oversized_document_policy != "skip" or "exceeds max_seq_length" not in str(error):
                    raise
                skipped_oversized_documents += 1
                if skipped_oversized_documents <= 10:
                    print(f"[WARN] skip oversized document from line {line_number}: {error}")
                continue
            if length > args.max_seq_length:
                raise ValueError(
                    f"Document from line {line_number} expands to {length} tokens, "
                    f"exceeding max_seq_length={args.max_seq_length}."
                )
            reached_document_limit = (
                args.max_documents_per_pack > 0 and len(current_documents) >= args.max_documents_per_pack
            )
            if current_documents and (current_tokens + length > args.max_seq_length or reached_document_limit):
                write_pack(output_stream, current_documents)
                pack_count += 1
                multi_document_pack_count += len(current_documents) > 1
                max_packed_tokens = max(max_packed_tokens, current_tokens)
                current_documents = []
                current_tokens = 0

            current_documents.append(document)
            current_tokens += length
            document_count += 1

        if current_documents:
            write_pack(output_stream, current_documents)
            pack_count += 1
            multi_document_pack_count += len(current_documents) > 1
            max_packed_tokens = max(max_packed_tokens, current_tokens)

    print("TND packing complete.")
    print(f"Documents: {document_count}")
    print(f"Packs: {pack_count}")
    print(f"Multi-document packs: {multi_document_pack_count}")
    print(f"Maximum packed tokens: {max_packed_tokens}")
    print(f"Skipped oversized documents: {skipped_oversized_documents}")
    print(f"Compression alignment: {alignment}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
