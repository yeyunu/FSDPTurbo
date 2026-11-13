from importlib import import_module
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from fsdp_turbo.quantization.mx_formats.mx_linear import _mx_linear_transform  # noqa: F401
    from fsdp_turbo.quantization.converter.mx_linear_converter import MXLinearConverter  # noqa: F401
    from fsdp_turbo.quantization.converter.mx_moe_converter import MXMoeConverter  # noqa: F401
    from fsdp_turbo.quantization.qat.w8a16.w8a16_converter import _make_w8a16_converter  # noqa: F401
    from fsdp_turbo.quantization.qat.w8a16.w8a16_linear import _w8a16_linear_transform  # noqa: F401
    from fsdp_turbo.quantization.qat.w4a16.w4a16_converter import _make_w4a16_converter  # noqa: F401
    from fsdp_turbo.quantization.qat.w4a16.w4a16_linear import _w4a16_linear_transform  # noqa: F401


_IMPORTS = {
    "_mx_linear_transform": "fsdp_turbo.quantization.mx_formats.mx_linear",
    "MXLinearConverter": "fsdp_turbo.quantization.converter.mx_linear_converter",
    "MXMoeConverter": "fsdp_turbo.quantization.converter.mx_moe_converter",
    "_make_w8a16_converter": "fsdp_turbo.quantization.qat.w8a16.w8a16_converter",
    "_w8a16_linear_transform": "fsdp_turbo.quantization.qat.w8a16.w8a16_linear",
    "_make_w4a16_converter": "fsdp_turbo.quantization.qat.w4a16.w4a16_converter",
    "_w4a16_linear_transform": "fsdp_turbo.quantization.qat.w4a16.w4a16_linear",
}

__all__ = list(_IMPORTS)


def __getattr__(name: str) -> Callable[..., object]:
    """Import the requested public quantization export only when accessed."""
    try:
        module_name = _IMPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
