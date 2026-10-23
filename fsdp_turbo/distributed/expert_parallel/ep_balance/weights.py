"""Per-forward duplicated weights and their gradient communication."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .ep_balance_strategy import EPBalanceStrategy


class EPBalanceWeights:
    """Prepare balanced weights without exposing executor details to dispatchers.

    Each forward owns its duplicate buffers. With no strategy the methods
    return the original tensors and perform no communication.
    """

    def __init__(self, strategy: "EPBalanceStrategy | None", gate_up_weights: torch.Tensor) -> None:
        """Start gate/up transfer before token dispatch or permutation."""
        self.strategy = strategy
        if strategy is not None:
            self.dup_experts_map = strategy.planner.get_plan_item("dup_experts_map")
            self.dup_gate_up_weights = strategy.executor.async_experts_param_comm(
                self.dup_experts_map, gate_up_weights, name="gate_up_weights"
            )

    def _append(self, local_weights: torch.Tensor, duplicate_weights: torch.Tensor, name: str) -> torch.Tensor:
        executor = self.strategy.executor
        # Hook a per-forward view, never the persistent parameter. Cat's hook
        # sends duplicate gradients before this view receives owned gradients.
        local_weights = local_weights.view_as(local_weights)
        executor.register_backward_dup_experts_grad_acc_hook(local_weights, name=name)
        weights = torch.cat([local_weights, duplicate_weights], dim=0)
        executor.register_backward_dup_experts_grad_comm_hook(self.dup_experts_map, weights, name=name)
        return weights

    def prepare_gate_up(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor) -> torch.Tensor:
        """Finish gate/up transfer and start down transfer before the first GEMM."""
        self.strategy.executor.wait_async_works_finished(name="gate_up_weights")
        gate_up_weights = self._append(gate_up_weights, self.dup_gate_up_weights, "gate_up_weights")
        self.dup_down_weights = self.strategy.executor.async_experts_param_comm(
            self.dup_experts_map, down_weights, name="down_weights"
        )
        return gate_up_weights

    def prepare_down(self, down_weights: torch.Tensor) -> torch.Tensor:
        """Finish down transfer after activation, immediately before the second GEMM."""
        if self.strategy is None:
            return down_weights
        self.strategy.executor.wait_async_works_finished(name="down_weights")
        return self._append(down_weights, self.dup_down_weights, "down_weights")
