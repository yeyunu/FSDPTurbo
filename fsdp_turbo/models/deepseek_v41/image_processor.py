"""DeepSeek-V4.1 image preprocessing.

Copied from the official ``inference/image_processor.py`` implementation and
adapted to accept in-memory PIL images from FSDPTurbo's multimodal collator.
"""

import base64
import io
import math
from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
import torch

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


@dataclass
class ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
):
    """Return the token grid produced from a resized image."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_image_tokens: int,
):
    """Find the largest aspect-preserving size within the image-token limit."""
    aspect_ratio = height / width
    max_width = math.sqrt((max_image_tokens - 2) / aspect_ratio + 0.25) - 0.5
    max_height = max_width * aspect_ratio
    cell_size = patch_size * downsample_ratio
    if max_width < 1.0:
        return (max_image_tokens - 2) // 2 * cell_size, cell_size
    if max_height < 1.0:
        return cell_size, (max_image_tokens - 3) * cell_size
    resize_ratio = min(
        math.floor(max_width) * cell_size / width,
        math.floor(max_height) * cell_size / height,
    )
    return (
        math.floor(height * resize_ratio / patch_size) * patch_size,
        math.floor(width * resize_ratio / patch_size) * patch_size,
    )


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_image_tokens: int,
):
    """Shrink an image until its visual span fits the configured token limit."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_image_tokens:
        best_height, best_width = solve_resize_ratio(
            height,
            width,
            patch_size,
            downsample_ratio,
            max_image_tokens,
        )
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_image_tokens
    return n_llm_h, n_llm_w, best_height, best_width


def load_image_bytes(image_record) -> bytes:
    """Load bytes from a PIL image or an official DeepSeek image record."""
    from PIL import Image

    if isinstance(image_record, Image.Image):
        image_buffer = io.BytesIO()
        image_record.save(image_buffer, format="PNG")
        return image_buffer.getvalue()
    if isinstance(image_record, str):
        image_record = {"url": image_record}
    elif isinstance(image_record, bytes):
        image_record = {"data": image_record}

    image_data = image_record.get("data")
    if isinstance(image_data, bytes):
        return image_data
    if isinstance(image_data, str):
        return base64.b64decode(image_data)

    image_source = image_record.get("source")
    if isinstance(image_source, dict):
        if image_source.get("data") is not None:
            return base64.b64decode(image_source["data"])
        if image_source.get("url"):
            return load_image_bytes({"url": image_source["url"]})

    image_url = image_record.get("url")
    if isinstance(image_url, str) and image_url:
        if image_url.startswith("data:"):
            header, _, payload = image_url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if image_url.startswith(("http://", "https://")):
            with urlopen(image_url, timeout=30) as response:
                return response.read()
        with open(image_url, "rb") as image_file:
            return image_file.read()

    raise ValueError(f"Cannot load image from record: {list(image_record.keys())}")


def plan_image_grid(width: int, height: int, model_args):
    """Return the DeepSeek resize and patch-grid plan for an image."""
    patch_size = model_args.vision_patch_size
    if model_args.vision_max_wh_ratio is not None and width > height * model_args.vision_max_wh_ratio:
        width = height * model_args.vision_max_wh_ratio
    if 0 < width * height < model_args.vision_min_pixels:
        resize_ratio = (model_args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * resize_ratio)
        height = int(height * resize_ratio)
    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    return safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        model_args.vision_downsample_ratio,
        model_args.vision_max_n_token,
    )


def load_image(image_record, model_args):
    """Load and transform one image into ViT patches."""
    from PIL import Image, ImageOps

    patch_size = model_args.vision_patch_size
    with Image.open(io.BytesIO(load_image_bytes(image_record))) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, model_args)
    n_vit_h, n_vit_w = best_height // patch_size, best_width // patch_size
    if model_args.vision_max_wh_ratio is not None and image.width >= model_args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    image_tensor = ((image_tensor - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        image_tensor.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Build the delimiter, patch, and row-separator token-type layout."""
    token_types = [IMAGE_START]
    token_types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    token_types.append(IMAGE_END)
    return torch.tensor(token_types, dtype=torch.int64)


def prepare_vl_inputs(prompt, images, tokenizer, model_args):
    """Expand image placeholders into token spans and processed image inputs."""
    image_token_id = model_args.image_token_id
    placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if placeholder_id is not None and placeholder_id != tokenizer.unk_token_id:
        if placeholder_id != image_token_id:
            raise ValueError(f"Image placeholder token id {placeholder_id} does not match {image_token_id}.")
    prompt_tokens = tokenizer.encode(prompt)
    placeholder_count = sum(token == image_token_id for token in prompt_tokens)
    if placeholder_count != len(images):
        raise ValueError(f"Found {placeholder_count} image tokens but got {len(images)} images.")
    if placeholder_count and not model_args.vision_enabled:
        raise ValueError("The model has no enabled vision tower.")

    tokens = []
    token_types = []
    image_inputs = []
    image_iterator = iter(images)
    for token in prompt_tokens:
        if token != image_token_id:
            tokens.append(token)
            token_types.append(TEXT)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(next(image_iterator), model_args)
        types = image_token_types(n_llm_h, n_llm_w)
        image_inputs.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types))
        tokens.extend([image_token_id] * types.numel())
        token_types.extend(types.tolist())
    return tokens, token_types, image_inputs or None


__all__ = [
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_PLACEHOLDER",
    "IMAGE_START",
    "ImageInput",
    "TEXT",
    "image_token_types",
    "load_image",
    "prepare_vl_inputs",
]
