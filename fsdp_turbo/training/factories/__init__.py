from fsdp_turbo.training.factories.model_factory import ModelFactory
from fsdp_turbo.training.factories.optimizer_factory import OptimizerFactory
from fsdp_turbo.training.factories.scheduler_factory import SchedulerFactory
from fsdp_turbo.training.factories.tokenizer_factory import TokenizerFactory

__all__ = ["ModelFactory", "OptimizerFactory", "SchedulerFactory", "TokenizerFactory"]


def _register_adamw_swap():
    from fsdp_turbo.memory.swap_optimizer import AdamWSwap

    def _create(params, **kwargs):
        # Pop swap-specific kwargs; pass the rest through to AdamW
        mem_fraction_static = kwargs.pop("mem_fraction_static", 0.9)
        state_keys = kwargs.pop("state_keys", None)
        return AdamWSwap(
            params,
            mem_fraction_static=mem_fraction_static,
            state_keys=state_keys,
            **kwargs,
        )

    OptimizerFactory.register("AdamWSwap", _create)


_register_adamw_swap()
