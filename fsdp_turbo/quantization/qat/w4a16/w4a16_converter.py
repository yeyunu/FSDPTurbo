from torch import nn


from fsdp_turbo.quantization.qat.w4a16.w4a16_config import get_w4a16_config
from fsdp_turbo.quantization.converter.model_converter import register_model_converter
from fsdp_turbo.quantization.qat.w4a16.w4a16_gmm import _w4a16_gmm_transform
from fsdp_turbo.quantization.qat.w4a16.w4a16_linear import _w4a16_linear_transform
from fsdp_turbo.quantization.utils import w4a16_module_filter_fn, w4a16_moe_filter_fn
from fsdp_turbo.fsdp_turbo_config import QuantizeConfig


def _make_w4a16_converter(name, filter_fn, handler=None):
    class _Converter:
        def __init__(self, config: QuantizeConfig):
            self.config = get_w4a16_config(config)

        def convert(self, model: nn.Module):
            from fsdp_turbo.quantization.quantize import quantize_

            quantize_(model, config=self.config, filter_fn=filter_fn, device=model.device, handler=handler)

    _Converter.__name__ = name.replace('.', '_')
    register_model_converter(_Converter, name)


_make_w4a16_converter("quantize.linear.w4a16", w4a16_module_filter_fn, handler=_w4a16_linear_transform)
_make_w4a16_converter("quantize.moe.w4a16", w4a16_moe_filter_fn, handler=_w4a16_gmm_transform)
