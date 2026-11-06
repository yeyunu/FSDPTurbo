"""DeepSeek-V4-Flash training adapter over the stock transformers deepseek_v4
implementation (transformers >= 5.8.1), with the FSDPTurbo expert-parallel
dispatch points wired in (see ``modeling.py``).
"""

from .adapter import (
    DeepseekV4CausalLMOutput,
    DeepseekV4ForCausalLMAdapter,
    build_deepseek_v4_model,
    build_deepseek_v4_model_args,
    build_deepseek_v4_model_meta,
    initialize_deepseek_v4_model,
    materialize_deepseek_v4_model,
    prepare_deepseek_v4_model_for_fsdp,
)
from .context_parallel import build_raw_attention_mask, deepseek_v4_cp_attention_forward
from .modeling import (
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4SparseMoeBlock,
)

__all__ = [
    "DeepseekV4CausalLMOutput",
    "DeepseekV4Config",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Experts",
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMAdapter",
    "DeepseekV4Model",
    "DeepseekV4SparseMoeBlock",
    "build_deepseek_v4_model",
    "build_deepseek_v4_model_args",
    "build_raw_attention_mask",
    "deepseek_v4_cp_attention_forward",
    "build_deepseek_v4_model_meta",
    "initialize_deepseek_v4_model",
    "materialize_deepseek_v4_model",
    "prepare_deepseek_v4_model_for_fsdp",
]
