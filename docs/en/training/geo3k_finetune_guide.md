# geo3k Dataset Fine-Tuning in Practice: Download, Processing, and Validation

## Overview

This document uses the geometry3k (geo3k) geometric reasoning dataset as an example to guide developers in preparing a small, out-of-the-box dataset for fine-tuning the Qwen3.5 multimodal model, covering the complete workflow from download and format conversion to data validation and starting training. The full geo3k dataset is about 41.5 MB (2,100+ image-text samples, with images embedded in the parquet file), three orders of magnitude smaller than LLaVA-Instruct-150K (a 229 MB conversation file plus 18 GB of COCO images), making it suitable for quickly validating the training pipeline and running small-scale experiments.

The main steps include:

- Environment preparation: install Python dependencies.
- Dataset download: obtain the geometry3k parquet file.
- Dataset processing: export the embedded images and convert to the FSDPTurbo multimodal format.
- Data validation: verify that the jsonl is loadable, the images are complete, and the format matches the target.
- Modify the training configuration: adjust the config for multimodal fine-tuning.

Prerequisites: FSDPTurbo is installed and the model weights are ready. For environment setup and model weight preparation, see [Quick Start: Pretraining and Fine-Tuning the Qwen3.5 Model](./quick_start_fsdpturbo.md).

## Environment Preparation

The conversion script depends on `datasets` and `pillow`. Install them in the virtual environment of the training environment:

```shell
pip install datasets pillow
```

Create a working directory:

```shell
cd /path/to/FSDPTurbo
mkdir -p dataset tools
```

## Dataset Download

The geometry3k dataset has been published on the Hugging Face Hub in parquet format. The train split is a single 41.5 MB file; simply download it with wget:

```shell
wget -O dataset/geometry3k_train.parquet \
  https://huggingface.co/datasets/hiyouga/geometry3k/resolve/main/data/train-00000-of-00001.parquet
```

If access to the official Hugging Face domain is restricted, replace the domain with hf-mirror.com.

After the download completes, confirm the file size:

```shell
ls -lh dataset/geometry3k_train.parquet
# Expected to be about 41.5M
```

The dataset schema is as follows:

| Column | Type | Description |
|---|---|---|
| images | list | Embedded binary images, one per sample |
| problem | str | Problem text, already containing the `<image>` placeholder |
| answer | str | Standard answer |

## Dataset Processing

### Conversion Script

Save the following content as `tools/convert_geo3k.py`. The script does two things: exports the images embedded in the parquet file as standalone PNG files, and assembles problem/answer into the messages conversation structure required by the FSDPTurbo multimodal branch.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_geo3k.py
geometry3k parquet -> FSDPTurbo multimodal fine-tuning jsonl
The output format is isomorphic to the geo3k_sft reference sample:
{"messages": [{"role": "user", "content": "<image>\\n..."},
               {"role": "assistant", "content": "..."}],
 "images": ["/abs/path/xxx.png"], "id": "geo3k_train_000000"}
"""

import argparse
import io
import json
import os

from datasets import load_dataset
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="geometry3k parquet -> FSDPTurbo jsonl")
    p.add_argument("--src", default="./dataset/geometry3k_train.parquet",
                   help="Input parquet path")
    p.add_argument("--img-dir", default="./dataset/geo3k_images",
                   help="Directory to export images to")
    p.add_argument("--dst", default="./dataset/train.jsonl",
                   help="Output jsonl path")
    return p.parse_args()


def normalize_problem(problem: str) -> str:
    """Remove existing placeholder forms and unify to a leading '<image>\\n'"""
    for pat in ("<image>\n", "\n<image>", "<image> ", "<image>"):
        problem = problem.replace(pat, "")
    return "<image>\n" + problem.strip()


def extract_image(im_obj):
    """Handle the various storage forms of the parquet images column"""
    if isinstance(im_obj, Image.Image):
        return im_obj
    if isinstance(im_obj, dict) and im_obj.get("bytes") is not None:
        return Image.open(io.BytesIO(im_obj["bytes"]))
    if isinstance(im_obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(im_obj))
    raise ValueError(f"Unrecognized image data type: {type(im_obj)}")


def main():
    args = parse_args()
    os.makedirs(args.img_dir, exist_ok=True)

    ds = load_dataset("parquet", data_files=args.src, split="train")
    print(ds)
    print("Column names:", ds.column_names)

    out, skip, exported = [], 0, 0
    for i, ex in enumerate(ds):
        # ---- 1. Export embedded images ----
        img_paths = []
        try:
            for j, im_obj in enumerate(ex["images"]):
                img = extract_image(im_obj)
                fname = f"geo3k_train_{i:06d}_{j}.png"
                fpath = os.path.abspath(os.path.join(args.img_dir, fname))
                img.convert("RGB").save(fpath)
                img_paths.append(fpath)
                exported += 1
        except Exception as e:
            print(f"[{i:06d}] Image export failed, skipping this sample: {e}")
            skip += 1
            continue

        # ---- 2. Assemble messages ----
        problem = (str(ex["problem"]) if ex["problem"] is not None else "").strip()
        answer = (str(ex["answer"]) if ex["answer"] is not None else "").strip()
        if not problem or not answer:
            skip += 1
            continue

        out.append({
            "messages": [
                {"role": "user", "content": normalize_problem(problem)},
                {"role": "assistant", "content": answer},
            ],
            "images": img_paths,
            "id": f"geo3k_train_{i:06d}",
        })

    # ---- 3. Write jsonl (one JSON object per line) ----
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone: wrote {len(out)} samples to {args.dst}")
    print(f"  Exported {exported} images to {args.img_dir}")
    print(f"  Skipped {skip} samples")
    if out:
        print("First sample preview:")
        print(json.dumps(out[0], ensure_ascii=False, indent=2)[:500])


if __name__ == "__main__":
    main()
```

### Running the Conversion

Run the following command in the FSDPTurbo root directory:

```shell
python tools/convert_geo3k.py \
  --src ./dataset/geometry3k_train.parquet \
  --img-dir ./dataset/geo3k_images \
  --dst ./dataset/train.jsonl
```

The expected end of the output:

```shell
Done: wrote 2101 samples to ./dataset/train.jsonl
  Exported 2101 images to ./dataset/geo3k_images
  Skipped 0 samples
```

Two artifacts are generated:

- `dataset/train.jsonl`: the conversation file for training
- `dataset/geo3k_images/*.png`: 2,101 standalone image files

## Data Validation

Be sure to complete the following two checks before starting training, to avoid errors in the dataloader stage after entering multi-device training.

### JSONL Loadability

Load the conversion output with `datasets` and check that the `messages` and `images` columns are complete:

```shell
python -c "
from datasets import load_dataset
ds = load_dataset('json', data_files='./dataset/train.jsonl', split='train')
print(ds)
print('column names:', ds.column_names)
assert 'messages' in ds.column_names, 'missing messages column'
assert 'images' in ds.column_names, 'missing images column'
print('first user message:', ds[0]['messages'][0])
print('first image path:', ds[0]['images'][0])
"
```

The expected column names are `['messages', 'images', 'id']`, and the first user message starts with `<image>\n`.

### Image Completeness

Open each image file referenced by the jsonl to confirm that none are corrupted:

```shell
python -c "
import json, os
from PIL import Image
bad = 0
for line in open('./dataset/train.jsonl'):
    for p in json.loads(line)['images']:
        try:
            Image.open(p).load()
        except Exception:
            bad += 1
print('Corrupted images:', bad)
assert bad == 0, 'corrupted images found, please re-run the conversion script'
"
```

### Format Comparison

The conversion output must be fully isomorphic to the following reference sample:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nFind $HK$."},
    {"role": "assistant", "content": "10"}
  ],
  "images": ["/abs/path/dataset/geo3k_images/geo3k_train_000020_0.png"],
  "id": "geo3k_train_000020"
}
```

Three format details must match the reference sample:

- The placeholder is `<image>\n` (a newline, not a space)
- `images` is a list of absolute paths (a single image is also wrapped in a list)
- Each line is a standalone JSON object (jsonl)

## Modifying the Training Configuration

Edit [examples/qwen3_5/config.yaml](../../../examples/qwen3_5/config.yaml). Compared with pretraining, the fine-tuning stage mainly adjusts the optimizer learning rate and the data configuration:

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

# --- Data configuration (multimodal conversation format) ---
data:
  dataset_path: "./dataset/train.jsonl"       # Points to the conversion output
  split: "train"
  is_multimodal: true                         # Multimodal branch switch
  conversation_column: "messages"             # Conversation column name, consistent with the jsonl field
  image_column: "images"                      # Image column name, consistent with the jsonl field
  batch_size: 1
  max_seq_length: 4096
  num_workers: 8
```

The model weight, distributed, and memory configurations can use the defaults in the Quick Start document. For how to launch training (`run.sh` / `torchrun`) and verify the training logs, see [Quick Start](./quick_start_fsdpturbo.md).

## FAQs

| Error keyword | Cause | Solution |
|---|---|---|
| DatasetGenerationError | The jsonl pointed to by `dataset_path` failed to load | Run the validation script in the "[jsonl Loadability](#jsonl-loadability)" section to locate the issue |
| KeyError: 'messages' | The jsonl field name does not match `conversation_column` | Change the jsonl or the config so that they are consistent |
| FileNotFoundError ... .png | The image path is invalid or the `image_column` configuration is wrong | Validate using the "[Image Completeness](#image-completeness)" section; make sure `image_column: "images"` |
| Expected value: [0, 8) | The number of processes does not match the number of devices | `GPUS_PER_NODE=8` or `--nproc_per_node=8` |
| OOM (Out Of Memory) | Insufficient parallelism or sequences that are too long | Increase `fully_shard_parallel_size`, or reduce `max_seq_length` or `batch_size` |
| Loss not decreasing | Incorrect data format or placeholder | Compare field by field against the reference sample in the "[Format Comparison](#format-comparison)" section |

## Appendix: Dataset Size Comparison (Selection Reference)

| Dataset | Download Size | Number of Samples | Image Processing | Suitable Scenarios |
|---|---|---|---|---|
| geometry3k (this document) | 41.5 MB | 2,101 | One-time export via script | Quick validation / small-scale experiments |
| LLaVA-Instruct-150K detail_23k | 20.5 MB + 18 GB (COCO) | 23K | No conversion needed | Larger-scale image-text fine-tuning |
| LLaVA-Instruct-150K full | 229 MB + 18 GB (COCO) | 150K | No conversion needed | Production image-text training |

After geo3k has been validated end-to-end, you can expand the training scale by downloading and merging the validation/test splits (5.9 MB + 11.8 MB) with the same process, or switch to another multimodal instruction dataset; the conversion script only requires adjusting the logic for reading the problem/answer fields.
