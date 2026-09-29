# geo3k数据集微调实战：下载、处理与验证

## 概述

本文档以geometry3k（geo3k）几何推理数据集为例，指导开发者为Qwen3.5多模态模型微调准备一份小体积、开箱即用的数据集，并完成从下载、格式转换、数据校验到启动训练的全流程。geo3k全量约41.5 MB（2100+条图文样本，图片内嵌于parquet），相比LLaVA-Instruct-150K（对话文件229 MB + COCO图片18 GB）体积小三个数量级，适合快速跑通训练链路和小规模实验。

主要步骤包括：

- 环境准备：安装Python依赖
- 数据集下载：获取geometry3k parquet文件
- 数据集处理：导出内嵌图片+转换为FSDPTurbo多模态格式
- 数据验证：校验jsonl可加载、图片完整、格式与目标一致
- 修改训练配置：调整config适配多模态微调

前置条件：FSDPTurbo已安装、模型权重已就绪。环境搭建与模型权重准备请参考[快速入门：Qwen3.5模型预训练及微调](./quick_start_fsdpturbo.md)。

## 环境准备

转换脚本依赖datasets与pillow，在训练环境所在虚拟环境中安装：

```shell
pip install datasets pillow
```

创建工作目录：

```shell
cd /path/to/FSDPTurbo
mkdir -p dataset tools
```

## 数据集下载

geometry3k数据集已在HuggingFace Hub发布为parquet格式，train split单文件41.5 MB，直接wget即可：

```shell
wget -O dataset/geometry3k_train.parquet \
  https://huggingface.co/datasets/hiyouga/geometry3k/resolve/main/data/train-00000-of-00001.parquet
```

若访问HuggingFace官方域名受限，将域名替换为hf-mirror.com。

下载完成后确认文件大小：

```shell
ls -lh dataset/geometry3k_train.parquet
# 预期约 41.5M
```

数据集schema如下：

| 列名 | 类型 | 说明 |
|---|---|---|
| images | list | 内嵌二进制图片，每条1张 |
| problem | str | 问题文本，已含`<image>`占位符 |
| answer | str | 标准答案 |

## 数据集处理

### 转换脚本

将以下内容保存为`tools/convert_geo3k.py`。脚本完成两件事：把parquet内嵌图片导出为独立PNG文件、把problem/answer组装成FSDPTurbo多模态分支要求的messages对话结构。

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_geo3k.py
geometry3k parquet -> FSDPTurbo 多模态微调 jsonl
输出格式与 geo3k_sft 参考样本同构:
{"messages": [{"role": "user", "content": "<image>\n..."},
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
                   help="输入 parquet 路径")
    p.add_argument("--img-dir", default="./dataset/geo3k_images",
                   help="导出图片的目录")
    p.add_argument("--dst", default="./dataset/train.jsonl",
                   help="输出 jsonl 路径")
    return p.parse_args()


def normalize_problem(problem: str) -> str:
    """去掉原有占位符写法，统一为前置 '<image>\\n'"""
    for pat in ("<image>\n", "\n<image>", "<image> ", "<image>"):
        problem = problem.replace(pat, "")
    return "<image>\n" + problem.strip()


def extract_image(im_obj):
    """兼容 parquet images 列的多种存储形式"""
    if isinstance(im_obj, Image.Image):
        return im_obj
    if isinstance(im_obj, dict) and im_obj.get("bytes") is not None:
        return Image.open(io.BytesIO(im_obj["bytes"]))
    if isinstance(im_obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(im_obj))
    raise ValueError(f"无法识别的图片数据类型: {type(im_obj)}")


def main():
    args = parse_args()
    os.makedirs(args.img_dir, exist_ok=True)

    ds = load_dataset("parquet", data_files=args.src, split="train")
    print(ds)
    print("列名:", ds.column_names)

    out, skip, exported = [], 0, 0
    for i, ex in enumerate(ds):
        # ---- 1. 导出内嵌图片 ----
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
            print(f"[{i:06d}] 图片导出失败, 跳过该条: {e}")
            skip += 1
            continue

        # ---- 2. 组装 messages ----
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

    # ---- 3. 写 jsonl（每行一个 JSON 对象）----
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n完成: 写入 {len(out)} 条到 {args.dst}")
    print(f"  导出图片 {exported} 张到 {args.img_dir}")
    print(f"  跳过 {skip} 条")
    if out:
        print("首条样本预览:")
        print(json.dumps(out[0], ensure_ascii=False, indent=2)[:500])


if __name__ == "__main__":
    main()
```

### 执行转换

在FSDPTurbo根目录下执行以下命令：

```shell
python tools/convert_geo3k.py \
  --src ./dataset/geometry3k_train.parquet \
  --img-dir ./dataset/geo3k_images \
  --dst ./dataset/train.jsonl
```

预期输出末尾：

```shell
完成: 写入 2101 条到 ./dataset/train.jsonl
  导出图片 2101 张到 ./dataset/geo3k_images
  跳过 0 条
```

生成两份产物：

- `dataset/train.jsonl`：训练用对话文件
- `dataset/geo3k_images/*.png`：2101张独立图片文件

## 数据验证

训练启动前务必完成两项校验，避免进入多卡后才在dataloader阶段报错。

### jsonl可加载性

使用datasets加载转换产物，校验messages与images列是否完整：

```shell
python -c "
from datasets import load_dataset
ds = load_dataset('json', data_files='./dataset/train.jsonl', split='train')
print(ds)
print('列名:', ds.column_names)
assert 'messages' in ds.column_names, '缺少 messages 列'
assert 'images' in ds.column_names, '缺少 images 列'
print('首条 user 消息:', ds[0]['messages'][0])
print('首条图片路径:', ds[0]['images'][0])
"
```

预期列名为`['messages', 'images', 'id']`，首条user消息以`<image>\n`开头。

### 图片完整性

逐条打开jsonl引用的图片文件，确认无损坏：

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
print('损坏图片数:', bad)
assert bad == 0, '存在损坏图片，请重新执行转换脚本'
"
```

### 格式对照

转换产物应与以下参考样本完全同构：

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

三个格式细节需与参考样本一致：

- 占位符为`<image>\n`（换行而非空格）
- `images`为绝对路径列表（单图也包在列表里）
- 每行一个独立JSON对象（jsonl）

## 修改训练配置

编辑[examples/qwen3_5/config.yaml](../../../examples/qwen3_5/config.yaml)，微调阶段与预训练相比主要调整优化器学习率与数据配置：

```yaml
# --- 优化器配置（微调学习率更小）---
optimizer:
  optimizer_type: "AdamW"
  lr: 1.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.0
  lr_scheduler_type: "cosine"
  min_lr: 1.0e-6
  clip_grad: 1.0

# --- 数据配置（多模态对话格式）---
data:
  dataset_path: "./dataset/train.jsonl"       # 指向转换产物
  split: "train"
  is_multimodal: true                         # 多模态分支开关
  conversation_column: "messages"             # 对话列名，与 jsonl 字段一致
  image_column: "images"                      # 图片列名，与 jsonl 字段一致
  batch_size: 1
  max_seq_length: 4096
  num_workers: 8
```

模型权重、分布式、内存配置沿用快速入门文档中的默认值即可。训练启动方式（`run.sh` / `torchrun`）与训练日志验证请参考[快速入门](./quick_start_fsdpturbo.md)。

## 常见问题

| 报错关键字 | 原因 | 处理方式 |
|---|---|---|
| DatasetGenerationError | `dataset_path`指向的jsonl加载失败 | 执行[“jsonl可加载性”](#jsonl可加载性)一节验证脚本定位 |
| KeyError: 'messages' | jsonl字段名与`conversation_column`不一致 | 改jsonl或改config，保持一致 |
| FileNotFoundError ... .png | 图片路径失效或`image_column`配置错误 | 执行[“图片完整性”](#图片完整性)一节校验；确认`image_column: "images"` |
| Expected value: [0, 8) | 进程数与卡数不匹配 | `GPUS_PER_NODE=8`或`--nproc_per_node=8` |
| OOM (Out Of Memory) | 并行度不足或序列过长 | 增大`fully_shard_parallel_size`、减小`max_seq_length`或`batch_size` |
| Loss不下降 | 数据格式或占位符不对 | 对照[“格式对照”](#格式对照)一节参考样本逐字段核对 |

## 附：数据集规模对比（选型参考）

| 数据集 | 下载量 | 样本数 | 图片处理 | 适用场景 |
|---|---|---|---|---|
| geometry3k（本文档） | 41.5 MB | 2101 | 脚本一次性导出 | 快速跑通/小规模实验 |
| LLaVA-Instruct-150K detail_23k | 20.5 MB + 18 GB (COCO) | 23K | 无需转换 | 更大规模图文微调 |
| LLaVA-Instruct-150K全量 | 229 MB + 18 GB (COCO) | 150K | 无需转换 | 正式图文训练 |

geo3k跑通后如需扩大训练规模，可将validation / test split（5.9 MB + 11.8 MB）以同样流程下载合并，或改用其他多模态指令数据集，转换脚本仅需调整problem/answer字段的取值逻辑。
