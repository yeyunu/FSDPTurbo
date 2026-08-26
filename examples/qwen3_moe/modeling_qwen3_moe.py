from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm

from fsdp_turbo.modules.rms_norm import RMSNorm


class FSDPTurboQwen3MoeRMSNorm(RMSNorm, Qwen3MoeRMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        RMSNorm.__init__(self, dim=hidden_size, eps=eps)


modeling_qwen3_moe.Qwen3MoeRMSNorm = FSDPTurboQwen3MoeRMSNorm
