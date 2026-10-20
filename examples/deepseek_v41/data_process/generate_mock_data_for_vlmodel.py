"""Generate repeatable DeepSeek-V4.1-Flash image/text training data.

Adapted from MindSpeed-MM's generate_mock_data_for_vlmodel.py:
https://gitcode.com/Ascend/MindSpeed-MM/blob/master/mindspeed_mm/fsdp/tools/data_tool/generate_mock_data_for_vlmodel.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerFast

from fsdp_turbo.models.deepseek.deepseek_v41 import build_deepseek_v41_model_args
from fsdp_turbo.models.deepseek.deepseek_v41.image_processor import IMAGE_PLACEHOLDER, num_image_tokens, plan_image_grid
from fsdp_turbo.models.deepseek.deepseek_v41.processor import DeepSeekV41Processor
from fsdp_turbo.training.dataloader import _render_multimodal_record


SEED_TEXT = "A moment frozen in the flow of time"


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Mock image/text data for DeepSeek-V4.1-Flash")
    parser.add_argument("--config_path", help="Explicit model architecture JSON file")
    parser.add_argument("--model_name_or_path", help="Model directory containing config.json")
    parser.add_argument("--tokenizer_path", required=True, help="Local DeepSeek-V4.1-Flash tokenizer directory")
    parser.add_argument("--pic_width", type=int, default=1024)
    parser.add_argument("--pic_height", type=int, default=1024)
    parser.add_argument("--num_pics", type=int, default=1)
    parser.add_argument(
        "--text_length", type=int, default=2048, help="Minimum token count of the repeated user and assistant text"
    )
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--save_dir", type=Path, default=Path("./data/mocked_vl_data"))
    parser.add_argument("--output_format", choices=("jsonl", "json"), default="jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    for name in ("pic_width", "pic_height", "text_length", "num_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_pics < 0:
        parser.error("--num_pics cannot be negative")
    return args


def make_record(num_pics, repeats, image_path):
    text = (SEED_TEXT + " ") * repeats
    image_prompt = "<image>\n" * num_pics
    return {
        "messages": [
            {"role": "user", "content": image_prompt + "Please describe the image. " + text},
            {"role": "assistant", "content": text},
        ],
        "images": [str(image_path)] * num_pics,
    }


def measure_record(record, tokenizer, processor, image_tokens):
    rendered = _render_multimodal_record(record["messages"], record["images"], processor, "images")
    repeated_text = record["messages"][1]["content"]
    text_tokens = 2 * len(tokenizer.encode(repeated_text, add_special_tokens=False))
    prompt_ids = tokenizer.encode(rendered["text"])
    placeholders = sum(token == processor.image_token_id for token in prompt_ids)
    if placeholders != len(record["images"]):
        raise ValueError(f"Expected {len(record['images'])} image tokens, got {placeholders}.")
    expanded_tokens = len(prompt_ids) + placeholders * (image_tokens - 1)
    return text_tokens, expanded_tokens


def fit_record(num_pics, image_path, tokenizer, processor, image_tokens, text_length):
    # Match MindSpeed-MM's repetition rule, then verify the actual token count.
    seed_tokens = max(len(tokenizer.encode(SEED_TEXT)), 1)
    repeats = text_length // (seed_tokens * 2) + 1
    while True:
        record = make_record(num_pics, repeats, image_path)
        text_tokens, expanded_tokens = measure_record(record, tokenizer, processor, image_tokens)
        if text_tokens >= text_length:
            return record, text_tokens, expanded_tokens
        repeats += 1


def main(argv=None):
    args = get_args(argv)
    model_args = build_deepseek_v41_model_args(config_path=args.config_path, model_name_or_path=args.model_name_or_path)
    if args.num_pics and not model_args.vision_enabled:
        raise ValueError("The DeepSeek-V4.1 model config has no enabled vision tower.")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    processor = DeepSeekV41Processor(tokenizer, model_args)
    token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if token_id != model_args.image_token_id:
        raise ValueError(
            f"The tokenizer's image token ID ({token_id}) does not match "
            f"the model config ({model_args.image_token_id})."
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    image_path = (args.save_dir / f"test_pic_h{args.pic_height}_w{args.pic_width}.jpg").resolve()
    if args.num_pics:
        random_array = np.random.default_rng(args.seed).integers(
            0, 256, (args.pic_height, args.pic_width, 3), dtype=np.uint8
        )
        Image.fromarray(random_array, "RGB").save(image_path)
        grid_h, grid_w, _, _ = plan_image_grid(args.pic_width, args.pic_height, model_args)
        image_tokens = num_image_tokens(grid_h, grid_w)
    else:
        image_tokens = 0

    record, text_tokens, expanded_tokens = fit_record(
        args.num_pics,
        image_path,
        tokenizer,
        processor,
        image_tokens,
        args.text_length,
    )
    output_path = args.save_dir / (f"mock_data_pic_num_{args.num_pics}_textlen_{args.text_length}.{args.output_format}")
    with output_path.open("w", encoding="utf-8") as stream:
        if args.output_format == "jsonl":
            line = json.dumps(record, ensure_ascii=False) + "\n"
            for _ in range(args.num_samples):
                stream.write(line)
        else:
            json.dump([record] * args.num_samples, stream, ensure_ascii=False)

    print(f"Dataset: {output_path.resolve()}")
    if args.num_pics:
        print(f"Image: {image_path}")
    print(f"Samples: {args.num_samples}; images/sample: {args.num_pics}")
    print(f"Text tokens: {text_tokens}; expanded sequence tokens: {expanded_tokens}")


if __name__ == "__main__":
    main()
