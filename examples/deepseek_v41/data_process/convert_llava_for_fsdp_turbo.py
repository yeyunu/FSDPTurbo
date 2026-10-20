#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    """
    支持两种输入：
    1. JSON Array: [ {...}, {...} ]
    2. JSONL: 每行一个 JSON
    """
    with path.open("r", encoding="utf-8") as f:
        first_non_space = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_space = ch
                break

        f.seek(0)

        if first_non_space == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array in {path}, got {type(data)}")
            yield from data
        else:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL line {line_no}: {e}") from e


def convert_role(from_role: str) -> Optional[str]:
    """
    LLaVA:
      human -> user
      gpt   -> assistant
    """
    if from_role == "human":
        return "user"
    if from_role == "gpt":
        return "assistant"
    if from_role == "system":
        return "system"

    # 不认识的角色，保守跳过
    return None


def convert_messages(conversations: Any) -> List[Dict[str, str]]:
    if isinstance(conversations, str):
        conversations = json.loads(conversations)

    if not isinstance(conversations, list):
        raise ValueError(f"conversations should be list, got {type(conversations)}")

    messages: List[Dict[str, str]] = []

    for idx, msg in enumerate(conversations):
        if not isinstance(msg, dict):
            continue

        # 兼容原始 LLaVA from/value
        if "from" in msg and "value" in msg:
            role = convert_role(str(msg.get("from", "")))
            content = msg.get("value", "")
        # 兼容已经是 role/content 的数据
        elif "role" in msg and "content" in msg:
            role = str(msg.get("role", ""))
            content = msg.get("content", "")
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
        else:
            continue

        if role not in {"user", "assistant", "system"}:
            continue

        if content is None:
            content = ""

        messages.append(
            {
                "role": role,
                "content": str(content),
            }
        )

    return messages


def convert_images(
    image_value: Any,
    image_root: Optional[Path],
    use_abs_path: bool,
) -> List[str]:
    """
    原始 LLaVA:
      image: "000000033471.jpg"

    转换后:
      images: ["/workspace/datasets/coco/train2017/000000033471.jpg"]
    """
    if image_value is None:
        return []

    if isinstance(image_value, str):
        image_value = image_value.strip()
        if not image_value:
            return []

        # 兼容极少数 image 字段本身是 JSON 字符串的情况
        if image_value.startswith("[") or image_value.startswith("{"):
            try:
                image_value = json.loads(image_value)
            except json.JSONDecodeError:
                image_value = [image_value]
        else:
            image_value = [image_value]

    elif isinstance(image_value, dict):
        image_value = [
            image_value.get("path")
            or image_value.get("image")
            or image_value.get("file_name")
            or image_value.get("filename")
        ]

    elif not isinstance(image_value, list):
        image_value = [image_value]

    images: List[str] = []

    for img in image_value:
        if img is None:
            continue

        if isinstance(img, dict):
            img = img.get("path") or img.get("image") or img.get("file_name") or img.get("filename")

        if not img:
            continue

        img_path = Path(str(img))

        if image_root is not None and not img_path.is_absolute():
            img_path = image_root / img_path

        if use_abs_path:
            img_path = img_path.resolve(strict=False)

        images.append(str(img_path))

    return images


def convert_one(
    example: Dict[str, Any],
    image_root: Optional[Path],
    use_abs_path: bool,
) -> Dict[str, Any]:
    if "conversations" not in example:
        raise KeyError(f"Missing conversations. keys={list(example.keys())}")

    messages = convert_messages(example["conversations"])
    images = convert_images(
        image_value=example.get("image"),
        image_root=image_root,
        use_abs_path=use_abs_path,
    )

    output = {
        "messages": messages,
        "images": images,
    }

    if "id" in example:
        output["id"] = example["id"]

    return output


def main():
    parser = argparse.ArgumentParser(description="Convert LLaVA original format to FSDPTurbo multimodal JSONL format.")

    parser.add_argument(
        "--input",
        required=True,
        help="Input LLaVA json/jsonl file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output jsonl file.",
    )
    parser.add_argument(
        "--image-root",
        default="/workspace/datasets/coco/train2017",
        help="Image root dir. Default: /workspace/datasets/coco/train2017",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Convert at most N input samples.",
    )
    parser.add_argument(
        "--check-image-exists",
        action="store_true",
        help="Check whether converted image paths exist.",
    )
    parser.add_argument(
        "--drop-missing-image",
        action="store_true",
        help="Drop samples with missing images. Only works with --check-image-exists.",
    )
    parser.add_argument(
        "--relative-image-path",
        action="store_true",
        help="Keep relative image paths instead of absolute paths.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    image_root = Path(args.image_root) if args.image_root else None
    use_abs_path = not args.relative_image_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0
    missing_image_samples = 0
    empty_message_samples = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for example in iter_json_records(input_path):
            if args.max_samples is not None and total >= args.max_samples:
                break

            total += 1

            try:
                item = convert_one(
                    example=example,
                    image_root=image_root,
                    use_abs_path=use_abs_path,
                )
            except Exception as e:
                skipped += 1
                print(f"[WARN] skip index={total - 1}, error={e}")
                continue

            if not item["messages"]:
                empty_message_samples += 1
                skipped += 1
                print(f"[WARN] skip index={total - 1}, id={item.get('id')}, empty messages")
                continue

            if args.check_image_exists:
                missing = [p for p in item["images"] if not Path(p).exists()]
                if missing:
                    missing_image_samples += 1
                    print(f"[WARN] missing image index={total - 1}, id={item.get('id')}, missing={missing[:3]}")
                    if args.drop_missing_image:
                        skipped += 1
                        continue

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print("Convert done.")
    print(f"Input samples: {total}")
    print(f"Written samples: {written}")
    print(f"Skipped samples: {skipped}")
    print(f"Empty message samples: {empty_message_samples}")
    print(f"Samples with missing image: {missing_image_samples}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
