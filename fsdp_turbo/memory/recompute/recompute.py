# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import inspect
import logging
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import checkpoint

from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match


logger = logging.getLogger(__name__)

CheckpointContextFn = Callable[[], tuple[AbstractContextManager, AbstractContextManager]]
CheckpointContextFnFactory = Callable[[Optional[nn.Module]], Optional[CheckpointContextFn]]


@dataclass
class RecomputePlanConfig:
    """Activation recompute plan: what to checkpoint and how.

    Code-side counterpart of the YAML ``memory.recompute_plan`` list. An entry
    either matches a module name (that module's forward is checkpointed) or is
    written as ``module.pattern.method_name`` (only that method is checkpointed).
    """

    apply_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "Module patterns or 'module.pattern.method_name' entries to recompute."},
    )
    flatten_inputs: bool = field(
        default=False,
        metadata={
            "help": "Flatten (args, kwargs) into positional leaves before checkpointing so every input tensor "
            "reaches save_for_backward (needed by tenants that swap saved activations)."
        },
    )


def recompute_modules(
    model: nn.Module,
    plan: List[str] | RecomputePlanConfig,
    context_fn_factory: Optional[CheckpointContextFnFactory] = None,
    reject_nested: bool = False,
) -> nn.Module:
    """Wire checkpoint-based recompute onto the modules or methods in ``plan``.

    Args:
        model: Model whose matched module forwards or methods are wrapped in place.
        plan: Module patterns, method entries, and optional input-flattening behavior.
        context_fn_factory: Called once per target at wiring time. Module targets pass their module;
            method targets pass ``None``. It returns a checkpoint context_fn or ``None``.
        reject_nested: Whether to reject ancestor/descendant module checkpoint coverage.

    Returns:
        The input model with checkpoint wrappers installed.

    Raises:
        RuntimeError: If a pattern has no matching module or a target method is not callable.
        ValueError: If a method entry is malformed or nested module coverage is rejected.
    """
    plan = _normalize_plan(plan)

    # Dispatch each entry: one that matches a module in the tree wraps that
    # module's forward; anything else is a module.pattern.method_name entry.
    module_entries = []
    method_entries = []
    for entry in plan.apply_modules:
        if any(module_name_match(entry, name) for name, _ in model.named_modules()):
            module_entries.append(entry)
        else:
            method_entries.append(entry)

    modules = get_recompute_modules(model, module_entries) if module_entries else []
    if reject_nested:
        _check_no_nested_checkpoints(modules)
        if method_entries:
            print_rank(
                logger.warning,
                "recompute does not support nested checkpoints: make sure no method in apply_modules "
                "runs inside another apply_modules checkpoint coverage",
            )

    for name, module in modules:
        print_rank(logger.info, f'Applying recompute to module: {name}')
        module.forward = recompute_wrapper(
            module.forward,
            context_fn_factory,
            plan.flatten_inputs,
            module,
        )
    wrapped_module_ids = {id(module) for _, module in modules}
    wrapped_methods = set()
    for pattern in method_entries:
        _apply_recompute_method(model, pattern, plan, context_fn_factory, wrapped_module_ids, wrapped_methods)

    return model


def _normalize_plan(plan: List[str] | RecomputePlanConfig) -> RecomputePlanConfig:
    if isinstance(plan, RecomputePlanConfig):
        return plan
    return RecomputePlanConfig(apply_modules=list(plan))


def _apply_recompute_method(model, pattern, plan, context_fn_factory, wrapped_module_ids, wrapped_methods):
    """Wrap a method on every module matching the prefix of *pattern*.

    'model.layers.{*}._attention_forward' -> module pattern 'model.layers.{*}',
    method '_attention_forward'; the wrapper is set on each matched instance.
    """
    module_pattern, _, method_name = pattern.rpartition('.')
    if not module_pattern or not method_name:
        raise ValueError(f'[Recompute] method entry must look like \'module.pattern.method_name\', got {pattern!r}')
    matched = False
    for name, module in model.named_modules():
        if not module_name_match(module_pattern, name):
            continue
        matched = True
        method_key = (id(module), method_name)
        if method_key in wrapped_methods:
            continue
        if method_name == 'forward' and id(module) in wrapped_module_ids:
            continue
        method = getattr(module, method_name, None)
        if method is None or not callable(method):
            raise RuntimeError(f'[Recompute] No callable method {method_name!r} on module {name!r}.')
        print_rank(logger.info, f'Applying recompute to method: {name}.{method_name}')
        setattr(module, method_name, recompute_wrapper(method, context_fn_factory, plan.flatten_inputs))
        wrapped_methods.add(method_key)
    if not matched:
        raise RuntimeError(f'[Recompute] No module named {module_pattern} (from {pattern!r}).')


def _check_no_nested_checkpoints(modules):
    """Reject checkpoint coverage that nests (one matched module is an ancestor
    of another): the inner zone would run inside two checkpoints, which
    replay-style context policies cannot handle. Reject at wiring time rather
    than degrading silently.
    """
    names = [name for name, _ in modules]
    for outer in names:
        prefix = outer + '.'
        for inner in names:
            if inner != outer and inner.startswith(prefix):
                raise ValueError(
                    "recompute does not support nested checkpoints: "
                    f"recompute_plan.apply_modules matches both '{outer}' and its "
                    f"descendant '{inner}'. Remove one of the overlapping patterns."
                )


def get_recompute_modules(model: nn.Module, specs: List[str]) -> List[tuple[str, nn.Module]]:
    """Return each module matching any pattern in ``specs`` exactly once.

    Args:
        model: Model whose named modules are searched.
        specs: Module name patterns.

    Returns:
        ``(name, module)`` pairs in first-match order.

    Raises:
        RuntimeError: If no module matches any pattern.
    """
    matched_modules = []
    matched_module_ids = set()
    for spec in specs:
        for name, module in model.named_modules():
            if id(module) in matched_module_ids:
                continue
            if module_name_match(spec, name):
                matched_module_ids.add(id(module))
                matched_modules.append((name, module))
    if len(matched_modules) == 0:
        raise RuntimeError(f'[Recompute] No module named {specs}.')
    return matched_modules


def _flatten_call(function, args, kwargs):
    """Flatten the full (args, kwargs) input tree into unique leaves plus a
    rebuild recipe (call-shape normalization, semantics-preserving).

    Non-reentrant checkpoint only routes top-level positional tensors through
    save_for_backward (saved_tensors_hooks); kwargs are captured by reference in
    the checkpoint frame and held until backward. Flattening every leaf
    (keyword-only / **kwargs contents / tensors inside containers included) into
    positional arguments moves them all into the save_for_backward channel so
    tenants like ActStash can swap them out. The shim inside the checkpoint
    rebuilds the original structure before calling the wrapped function, so the
    function sees exactly the call it was invoked with.

    Leaves are deduplicated by identity: an object aliased across several slots
    occupies one leaf position, so recompute restores the alias (same object)
    instead of silently forking it into independent copies, and the saved
    channel packs it once instead of once per slot.
    """
    flat_leaves, tree_spec = tree_flatten((args, kwargs))
    unique_leaves, leaf_index, leaf_slots = [], {}, []
    for leaf in flat_leaves:
        key = id(leaf)
        if key not in leaf_index:
            leaf_index[key] = len(unique_leaves)
            unique_leaves.append(leaf)
        leaf_slots.append(leaf_index[key])

    def flattened_function(*leaves):
        rebuild = [leaves[slot] for slot in leaf_slots]
        original_args, original_kwargs = tree_unflatten(rebuild, tree_spec)
        return function(*original_args, **original_kwargs)

    return flattened_function, unique_leaves


def compose_checkpoint_contexts(context_fn_factories: List[CheckpointContextFn]) -> CheckpointContextFn:
    """Combine checkpoint context factories into one context_fn.

    Args:
        context_fn_factories: Factories invoked on every checkpoint execution.

    Returns:
        A context_fn whose contexts are entered in list order and exited in reverse order.
        Already-entered contexts are rolled back if a later context fails to enter.
    """

    def composed():
        contexts = [factory() for factory in context_fn_factories]
        if not contexts:
            return nullcontext(), nullcontext()
        forward_contexts, recompute_contexts = zip(*contexts)
        return _ComposedContext(forward_contexts), _ComposedContext(recompute_contexts)

    return composed


class _ComposedContext:
    def __init__(self, contexts):
        self._contexts = contexts

    def __enter__(self):
        self._stack = ExitStack()
        try:
            for context in self._contexts:
                self._stack.enter_context(context)
        except BaseException as error:
            self._stack.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def __exit__(self, *exc_info):
        return self._stack.__exit__(*exc_info)


def recompute_wrapper(
    function: Callable[..., object],
    context_fn_factory: Optional[CheckpointContextFnFactory] = None,
    flatten_inputs: bool = False,
    module: Optional[nn.Module] = None,
) -> Callable[..., object]:
    """Wrap a callable with non-reentrant checkpointing.

    Args:
        function: Module forward or method to wrap.
        context_fn_factory: Optional factory invoked once at wiring time.
        flatten_inputs: Whether to flatten the full argument tree before checkpointing.
        module: Owning module passed to the factory, or ``None`` for method targets.

    Returns:
        A wrapper that invokes ``function`` through non-reentrant checkpointing.
    """
    # Only inject the transformers-style cache kwarg when the wrapped forward
    # actually accepts it. Native Wan blocks do not take this argument.
    signature = inspect.signature(function)
    has_past_key_values = 'past_key_values' in signature.parameters

    context_fn = None
    if context_fn_factory is not None:
        context_fn = context_fn_factory(module)

    def wrapper(*args, **kwargs):
        if 'past_key_values' in kwargs:
            kwargs['past_key_values'] = None
        if has_past_key_values:
            bound = signature.bind(*args, **kwargs)
            bound.arguments['past_key_values'] = None
            args, kwargs = bound.args, bound.kwargs

        checkpoint_kwargs = {} if context_fn is None else {'context_fn': context_fn}
        if flatten_inputs:
            flattened_function, unique_leaves = _flatten_call(function, args, kwargs)
            return checkpoint(flattened_function, *unique_leaves, use_reentrant=False, **checkpoint_kwargs)
        return checkpoint(function, *args, use_reentrant=False, **checkpoint_kwargs, **kwargs)

    return wrapper
