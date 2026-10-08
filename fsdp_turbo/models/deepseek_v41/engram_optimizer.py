"""Dense + active-row Adam training utilities for DeepSeek-V4.1 Host Engram."""

from __future__ import annotations

import torch


class ActiveRowAdam(torch.optim.Optimizer):
    """SparseAdam-equivalent updates restricted to active embedding rows.

    Host Engram weights are BF16 while both moments and pending sparse
    gradients are FP32. Native SparseAdam eventually performs a mixed-dtype
    ``BF16 dense += FP32 sparse`` operation whose CPU cost scales with the
    complete table. This implementation gathers, updates, and writes back only
    the rows named by the coalesced sparse gradient.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    def initialize_state(self, parameter):
        state = self.state[parameter]
        if state:
            return state
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
        return state

    @torch.no_grad()
    def step(self, closure=None, *, sparse_gradients=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        if sparse_gradients is not None:
            sparse_gradients = tuple(sparse_gradients)
            if len(sparse_gradients) != len(parameters):
                raise ValueError("ActiveRowAdam requires one sparse gradient per parameter.")

        gradient_index = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad if sparse_gradients is None else sparse_gradients[gradient_index]
                gradient_index += 1
                if gradient is None:
                    continue
                if not gradient.is_sparse:
                    raise RuntimeError("ActiveRowAdam requires sparse COO gradients.")
                if gradient.shape != parameter.shape:
                    raise RuntimeError("ActiveRowAdam gradient and parameter shapes must match.")
                gradient = gradient.coalesce()
                indices = gradient.indices()
                if indices.shape[0] != 1:
                    raise RuntimeError("ActiveRowAdam supports row-sparse embedding gradients only.")

                state = self.initialize_state(parameter)
                state["step"] += 1
                if gradient._nnz() == 0:
                    continue

                row_ids = indices[0]
                gradient_rows = gradient.values().float()
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                active_avg = exp_avg.index_select(0, row_ids)
                active_avg_sq = exp_avg_sq.index_select(0, row_ids)
                active_avg.mul_(beta1).add_(gradient_rows, alpha=1 - beta1)
                active_avg_sq.mul_(beta2).addcmul_(gradient_rows, gradient_rows, value=1 - beta2)
                exp_avg.index_copy_(0, row_ids, active_avg)
                exp_avg_sq.index_copy_(0, row_ids, active_avg_sq)

                step = state["step"]
                step_size = group["lr"] * (1 - beta2**step) ** 0.5 / (1 - beta1**step)
                update = active_avg / (active_avg_sq.sqrt() + group["eps"])
                active_weight = parameter.index_select(0, row_ids).float()
                active_weight.add_(update, alpha=-step_size)
                parameter.index_copy_(0, row_ids, active_weight.to(dtype=parameter.dtype))
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Base Optimizer restoration follows the BF16 parameter dtype. Keep
        # moments FP32 even when this optimizer is loaded directly by DCP.
        for state in self.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    state[name] = state[name].float()


class EngramOptimizer:
    """Present dense and Host active-row optimizers as one component."""

    def __init__(self, model, tables, dense_optimizer_type, **kwargs):
        from fsdp_turbo.training.factories.optimizer_factory import (
            OptimizerFactory,
        )

        if dense_optimizer_type == "EngramOptimizer":
            raise ValueError("EngramOptimizer cannot use itself as its dense optimizer.")

        tables = tuple(tables)
        sparse_parameters = [table.weight for table in tables]
        sparse_parameter_ids = {id(parameter) for parameter in sparse_parameters}
        dense_parameters = [parameter for parameter in model.parameters() if id(parameter) not in sparse_parameter_ids]
        dense_optimizer = OptimizerFactory.create(
            dense_optimizer_type,
            dense_parameters,
            **kwargs,
        )
        sparse_optimizer = OptimizerFactory.create(
            "ActiveRowAdam",
            sparse_parameters,
            lr=kwargs.get("lr", 1e-3),
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )

        self.dense_optimizer = dense_optimizer
        self.sparse_optimizer = sparse_optimizer
        self.tables = tables
        self.checkpoint_optimizers = (dense_optimizer, sparse_optimizer)

    @property
    def param_groups(self):
        return self.dense_optimizer.param_groups + self.sparse_optimizer.param_groups

    def step(self, closure=None):
        dense_result = self.dense_optimizer.step(closure=closure)
        self.sparse_optimizer.step(sparse_gradients=[table.pending_sparse_grad() for table in self.tables])
        for table in self.tables:
            table.mark_sparse_step_complete()
        return dense_result

    def zero_grad(self, set_to_none=True):
        self.dense_optimizer.zero_grad(set_to_none=set_to_none)
        self.sparse_optimizer.zero_grad(set_to_none=set_to_none)
        for table in self.tables:
            table.clear_sparse_gradient()

    def state_dict(self):
        return {
            "dense": self.dense_optimizer.state_dict(),
            "engram_sparse": self.sparse_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.dense_optimizer.load_state_dict(state_dict["dense"])
        self.sparse_optimizer.load_state_dict(state_dict["engram_sparse"])
        # Optimizer.load_state_dict casts floating state to the BF16 parameter
        # dtype. Restore the configured FP32 moment representation.
        for state in self.sparse_optimizer.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    state[name] = state[name].float()

    def _ensure_fp32_sparse_state(self, parameter):
        self.sparse_optimizer.initialize_state(parameter)

    def prepare_checkpoint(self):
        """Materialize active-row Adam's lazy state for a stable save schema."""
        for group in self.sparse_optimizer.param_groups:
            for parameter in group["params"]:
                state = self.sparse_optimizer.state[parameter]
                if state:
                    continue
                self._ensure_fp32_sparse_state(parameter)

    def dcp_state_dict(self, model):
        result = {}
        save_engram_checkpoint_state(result, model, self)
        return result

    def load_dcp_state_dict(self, state_dict, model, *, strict):
        load_engram_checkpoint_state(state_dict, model, self, strict=strict)


class EngramSchedulerContainer:
    def __init__(self, schedulers):
        self.schedulers = tuple(schedulers)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        return [scheduler.state_dict() for scheduler in self.schedulers]

    def load_state_dict(self, state_dict):
        for scheduler, scheduler_state in zip(self.schedulers, state_dict, strict=True):
            scheduler.load_state_dict(scheduler_state)


def save_engram_checkpoint_state(result, model, optimizer) -> None:
    """Save owner-local Host Engram state separately from canonical DCP state."""
    from torch.distributed.checkpoint.state_dict import get_state_dict

    optimizer.prepare_checkpoint()
    saved_weights = [(table, table._parameters.pop("weight")) for table in optimizer.tables]
    try:
        model_state, optimizer_state = get_state_dict(model, optimizer.dense_optimizer)
    finally:
        for table, weight in saved_weights:
            table.register_parameter("weight", weight)

    result["model"] = model_state
    result["optim"] = optimizer_state
    result["engram_tables"] = {
        f"layer_{table.layer_id}{table._checkpoint_suffix()}": table.weight.detach() for table in optimizer.tables
    }
    result["engram_metadata"] = {
        f"layer_{table.layer_id}{table._checkpoint_suffix()}": {
            "logical_num_embeddings": table.logical_num_embeddings,
            "physical_num_embeddings": table.num_embeddings,
            "embedding_dim": table.embedding_dim,
            "hash_fingerprint": getattr(table, "hash_fingerprint", ""),
        }
        for table in optimizer.tables
    }
    suffix = optimizer.tables[0]._checkpoint_suffix()
    result["engram_optimizer"] = {f"sparse_adam{suffix}": optimizer.sparse_optimizer.state_dict()}


def load_engram_checkpoint_state(state_dict, model, optimizer, *, strict) -> None:
    """Restore canonical DCP state and owner-local Host Engram shards."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_state_dict,
    )

    saved_weights = [(table, table._parameters.pop("weight")) for table in optimizer.tables]
    try:
        set_state_dict(
            model,
            optimizer.dense_optimizer,
            model_state_dict=state_dict.get("model", {}),
            optim_state_dict=state_dict.get("optim", {}),
            options=StateDictOptions(strict=strict),
        )
    finally:
        for table, weight in saved_weights:
            table.register_parameter("weight", weight)

    table_state = state_dict.get("engram_tables", {})
    table_metadata = state_dict.get("engram_metadata", {})
    with torch.no_grad():
        for table in optimizer.tables:
            key = f"layer_{table.layer_id}{table._checkpoint_suffix()}"
            expected = {
                "logical_num_embeddings": table.logical_num_embeddings,
                "physical_num_embeddings": table.num_embeddings,
                "embedding_dim": table.embedding_dim,
                "hash_fingerprint": getattr(table, "hash_fingerprint", ""),
            }
            if table_metadata.get(key) != expected:
                raise ValueError(
                    f"Engram checkpoint metadata does not match {key}: {table_metadata.get(key)} != {expected}."
                )
            table.weight.copy_(table_state[key])
            if table._buffer is not None:
                table._register_storage()

    suffix = optimizer.tables[0]._checkpoint_suffix()
    sparse_state = state_dict.get("engram_optimizer", {})
    optimizer.sparse_optimizer.load_state_dict(sparse_state[f"sparse_adam{suffix}"])


def register_engram_optimizers() -> None:
    """Register model-owned optimizers without coupling the generic factory to DeepSeek."""
    from fsdp_turbo.training.factories.optimizer_factory import OptimizerFactory

    OptimizerFactory.register("ActiveRowAdam", ActiveRowAdam)
    OptimizerFactory.register("EngramOptimizer", EngramOptimizer)


register_engram_optimizers()


__all__ = [
    "ActiveRowAdam",
    "EngramOptimizer",
    "EngramSchedulerContainer",
    "load_engram_checkpoint_state",
    "register_engram_optimizers",
    "save_engram_checkpoint_state",
]
