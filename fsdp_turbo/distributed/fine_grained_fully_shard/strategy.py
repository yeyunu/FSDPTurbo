# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from abc import ABC, abstractmethod
import inspect
import logging
from typing import Callable, Optional

from torch import nn

from fsdp_turbo.utils.log import log_warning_once


logger = logging.getLogger(__name__)


def create_fully_shard_wrapper(fully_shard_fn: Callable, supports_hook_module: bool) -> Callable:
    """
    Create a unified fully_shard wrapper to handle parameter differences.

    Args:
        fully_shard_fn: The original fully_shard function.
        supports_hook_module: Whether the function supports hook_module parameter.

    Returns:
        A wrapped fully_shard function that uniformly accepts hook_module parameter.
    """

    def wrapper(module: nn.Module, hook_module: Optional[nn.Module] = None, **kwargs):
        if supports_hook_module:
            return fully_shard_fn(module, hook_module=hook_module, **kwargs)
        else:
            return fully_shard_fn(module, **kwargs)

    return wrapper


class FSDPStrategy(ABC):
    """Base class for FSDP implementation strategies."""

    @abstractmethod
    def get_fully_shard_fn(self) -> Callable:
        """Return the original fully_shard function."""
        pass

    @property
    @abstractmethod
    def supports_hook_module(self) -> bool:
        """Whether the implementation supports hook_module parameter."""
        pass

    def get_unified_fully_shard_fn(self) -> Callable:
        """Return a unified fully_shard wrapper function."""
        return create_fully_shard_wrapper(self.get_fully_shard_fn(), self.supports_hook_module)


class CustomFSDPStrategy(FSDPStrategy):
    """FSDPTurbo custom FSDP implementation."""

    def get_fully_shard_fn(self) -> Callable:
        from fsdp_turbo.distributed.fine_grained_fully_shard import fully_shard

        return fully_shard

    @property
    def supports_hook_module(self) -> bool:
        return True


class NativeFSDPStrategy(FSDPStrategy):
    """PyTorch native FSDP implementation."""

    def get_fully_shard_fn(self) -> Callable:
        from torch.distributed.fsdp import fully_shard

        return fully_shard

    @property
    def supports_hook_module(self) -> bool:
        """Whether the active PyTorch FSDP function accepts a hook module."""
        fully_shard_fn = self.get_fully_shard_fn()
        try:
            parameters = inspect.signature(fully_shard_fn, follow_wrapped=False).parameters
            hook_parameter = parameters.get("hook_module")
            if hook_parameter is None and any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            ):
                hook_parameter = inspect.signature(fully_shard_fn).parameters.get("hook_module")
            return hook_parameter is not None and hook_parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        except (TypeError, ValueError):
            log_warning_once(
                logger,
                "Cannot inspect the active fully_shard signature; hook_module will not be passed. "
                "Nested FSDP with activation recomputation requires an implementation that exposes hook_module.",
            )
            return False


def get_fsdp_strategy(implementation: str) -> FSDPStrategy:
    """
    Get FSDP strategy based on configuration.

    Args:
        implementation: Implementation type, either 'custom' or 'native'.

    Returns:
        Corresponding FSDP strategy instance.
    """
    strategies = {
        'custom': CustomFSDPStrategy(),
        'native': NativeFSDPStrategy(),
    }
    if implementation not in strategies:
        raise ValueError(f"Unknown FSDP implementation: {implementation}. Supported: {list(strategies.keys())}")
    return strategies[implementation]
