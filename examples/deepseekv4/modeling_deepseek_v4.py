from transformers.models.deepseek_v4 import modeling_deepseek_v4
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RMSNorm, DeepseekV4HyperConnection

from fsdp_turbo.modules.rms_norm import RMSNorm
from fsdp_turbo.modules.hyper_connection import HyperConnection


class FSDPTurboDeepSeekV4RMSNorm(RMSNorm, DeepseekV4RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        RMSNorm.__init__(self, dim=hidden_size, eps=eps)


class FSDPTurboDeepSeekV4HyperConnection(HyperConnection, DeepseekV4HyperConnection):
    def __init__(self, config):
        HyperConnection.__init__(self, config)


modeling_deepseek_v4.DeepseekV4RMSNorm = FSDPTurboDeepSeekV4RMSNorm
modeling_deepseek_v4.DeepseekV4HyperConnection = FSDPTurboDeepSeekV4HyperConnection
