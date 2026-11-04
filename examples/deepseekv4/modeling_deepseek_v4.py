import torch
import torch.nn.functional as F

from torch import nn
from transformers.models.deepseek_v4 import modeling_deepseek_v4
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RMSNorm, DeepseekV4HyperConnection, repeat_kv

from fsdp_turbo.modules.rms_norm import RMSNorm
from fsdp_turbo.ops.rms_norm import rms_norm
from fsdp_turbo.modules.hyper_connection import HyperConnection


class FSDPTurboDeepSeekV4RMSNorm(RMSNorm, DeepseekV4RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        RMSNorm.__init__(self, dim=hidden_size, eps=eps)

    def forward(self, x):
        weight = self.weight.to(dtype=x.dtype)
        return rms_norm(x, weight, self.eps)


class FSDPTurboDeepSeekV4HyperConnection(HyperConnection, DeepseekV4HyperConnection):
    def __init__(self, config):
        HyperConnection.__init__(self, config)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    if query.size(1) != key.size(1):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
    else:
        key_states = key
        value_states = value

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # Preserve causal, sliding-window and compressed-KV visibility supplied by
    # the model. Ignoring this mask leaks future tokens during actor training.
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # The compressor CP wrapper owns head sharding; eager consumes local sinks directly.
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)

    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    del attn_weights

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    del combined_logits
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


modeling_deepseek_v4.DeepseekV4RMSNorm = FSDPTurboDeepSeekV4RMSNorm
modeling_deepseek_v4.DeepseekV4HyperConnection = FSDPTurboDeepSeekV4HyperConnection
modeling_deepseek_v4.eager_attention_forward = eager_attention_forward
