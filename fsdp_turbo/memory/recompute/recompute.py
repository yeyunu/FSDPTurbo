# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from contextlib import nullcontext

from torch.utils.checkpoint import checkpoint

from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match
from fsdp_turbo.quantization.core.recompute_phase import mark_module_for_recompute, mxfp8_recompute_context

logger = logging.getLogger(__name__)


def recompute_modules(model, plan):
    modules = get_recompute_modules(model, plan)

    for name, module in modules:
        print_rank(logger.info, f'Applying recompute to module: {name}')
        mark_module_for_recompute(module)
        module.forward = recompute_wrapper(module, module.forward)

    return model


def get_recompute_modules(modules, plan):
    matched_modules = []
    for plan_name in plan:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                matched_modules.append((name, module))
    if len(matched_modules) == 0:
        raise RuntimeError(f'[Recompute] No module named {plan}.')
    return matched_modules


def recompute_wrapper(module, function):
    def wrapper(*args, **kwargs):
        # transformers kv cache must be set None, or model use_cache=False
        if 'past_key_values' in kwargs:
            kwargs['past_key_values'] = None

        def context_fn():
            return nullcontext(), mxfp8_recompute_context(module)

        return checkpoint(function, *args, use_reentrant=False, context_fn=context_fn, **kwargs)

    return wrapper
