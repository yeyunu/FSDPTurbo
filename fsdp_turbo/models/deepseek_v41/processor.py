"""Training processor for DeepSeek-V4.1 interleaved text and images."""

from collections.abc import Sequence

import torch

from .image_processor import IMAGE_PLACEHOLDER, TEXT, prepare_vl_inputs

BEGIN_OF_SENTENCE = "<｜begin▁of▁sentence｜>"
END_OF_SENTENCE = "<｜end▁of▁sentence｜>"
SYSTEM_TOKEN = "<｜System｜>"  # nosec B105 - model vocabulary marker, not a credential
USER_TOKEN = "<｜User｜>"  # nosec B105 - model vocabulary marker, not a credential
ASSISTANT_TOKEN = "<｜Assistant｜>"  # nosec B105 - model vocabulary marker, not a credential
THINKING_END_TOKEN = "</think>"  # nosec B105 - model vocabulary marker, not a credential
IMAGE_TOKEN = "<｜deepseek_image｜>"  # nosec B105 - model vocabulary marker, not a credential


def prepare_deepseek_v41_tokenizer(tokenizer, model_args) -> bool:
    """Ensure the validation tokenizer uses the model's fixed image-token ID.

    Returns ``True`` when reserved tokens had to be added. Callers use this to
    disable Engram for the synthetic validation vocabulary.
    """
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if image_token_id is None:
        missing_before_image = model_args.image_token_id - len(tokenizer)
        if missing_before_image < 0:
            raise ValueError(
                "Tokenizer is missing the image token but its vocabulary already "
                "extends beyond the configured image_token_id."
            )
        reserved = [f"<｜reserved_image_prefix_{index}｜>" for index in range(missing_before_image)]
        tokenizer.add_special_tokens({"additional_special_tokens": reserved + [IMAGE_TOKEN]})
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        added_tokens = True
    else:
        added_tokens = False
    if image_token_id != model_args.image_token_id:
        raise ValueError(f"Image token id mismatch: tokenizer={image_token_id}, model={model_args.image_token_id}.")
    return added_tokens


def _render_content(content) -> str:
    if isinstance(content, str):
        return content
    rendered_parts = []
    for content_part in content or []:
        if not isinstance(content_part, dict):
            continue
        if content_part.get("type") in {"image", "image_url"}:
            rendered_parts.append(IMAGE_PLACEHOLDER)
        elif content_part.get("type") == "text":
            rendered_parts.append(content_part.get("text", ""))
    return "\n\n".join(rendered_parts)


def encode_chat_messages(messages: Sequence[dict]) -> str:
    """Encode common system/user/assistant SFT messages with V4.1 tokens."""
    prompt_parts = [BEGIN_OF_SENTENCE]
    for message_index, message in enumerate(messages):
        role = message.get("role")
        content = _render_content(message.get("content", ""))
        if role == "system":
            prompt_parts.append(SYSTEM_TOKEN)
            prompt_parts.append(content)
        elif role == "user":
            prompt_parts.append(USER_TOKEN)
            prompt_parts.append(content)
            next_role = messages[message_index + 1].get("role") if message_index + 1 < len(messages) else None
            if next_role in {None, "assistant"}:
                prompt_parts.extend((ASSISTANT_TOKEN, THINKING_END_TOKEN))
        elif role == "assistant":
            prompt_parts.extend((content, END_OF_SENTENCE))
        else:
            raise ValueError(f"Unsupported DeepSeek-V4.1 training role: {role!r}.")
    return "".join(prompt_parts)


class DeepSeekV41Processor:
    """Convert chat records and images into the reference model's VL inputs."""

    def __init__(self, tokenizer, model_args):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.image_token_id = model_args.image_token_id

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        del add_generation_prompt
        prompt = encode_chat_messages(messages)
        return self.tokenizer.encode(prompt) if tokenize else prompt

    def __call__(
        self,
        *,
        text,
        images=None,
        padding="longest",
        truncation=True,
        max_length=None,
        return_tensors="pt",
        **processor_inputs,
    ):
        del processor_inputs
        if return_tensors != "pt":
            raise ValueError("DeepSeekV41Processor only supports return_tensors='pt'.")
        prompts = [text] if isinstance(text, str) else list(text)
        flattened_images = list(images or [])
        image_offset = 0
        token_sequences = []
        token_type_sequences = []
        batch_image_inputs = []

        for prompt in prompts:
            image_count = prompt.count(IMAGE_PLACEHOLDER)
            prompt_images = flattened_images[image_offset : image_offset + image_count]
            image_offset += image_count
            tokens, token_types, image_inputs = prepare_vl_inputs(
                prompt,
                prompt_images,
                self.tokenizer,
                self.model_args,
            )
            if truncation and max_length is not None and len(tokens) > max_length:
                raise ValueError(
                    "DeepSeek-V4.1 multimodal sample exceeds max_seq_length after "
                    "image-span expansion; truncating an image span is unsafe."
                )
            token_sequences.append(tokens)
            token_type_sequences.append(token_types)
            batch_image_inputs.append(image_inputs)

        if image_offset != len(flattened_images):
            raise ValueError(
                f"Prompts contain {image_offset} image placeholders but received {len(flattened_images)} images."
            )
        if padding != "longest":
            raise ValueError("DeepSeekV41Processor currently requires padding='longest'.")

        padded_length = max(len(tokens) for tokens in token_sequences)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id or eos_token_id.")

        input_ids = []
        attention_masks = []
        token_types = []
        for tokens, sequence_token_types in zip(token_sequences, token_type_sequences):
            padding_length = padded_length - len(tokens)
            input_ids.append(tokens + [pad_token_id] * padding_length)
            attention_masks.append([1] * len(tokens) + [0] * padding_length)
            token_types.append(sequence_token_types + [TEXT] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "token_types": torch.tensor(token_types, dtype=torch.long),
            "images": batch_image_inputs,
        }


__all__ = [
    "DeepSeekV41Processor",
    "encode_chat_messages",
    "prepare_deepseek_v41_tokenizer",
]
