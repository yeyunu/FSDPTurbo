"""Load-balancing setup and routing shared by EP dispatchers."""

import torch
import torch.distributed as dist

from fsdp_turbo.distributed.expert_parallel.comm import allgather_tokens_in_ep
from fsdp_turbo.distributed.expert_parallel.utils import get_num_global_tokens_per_local_expert
from fsdp_turbo.distributed.expert_parallel.ep_balance.training_context import TrainingContext, TrainingStage
from .greedy_dup_experts_executor import DupExpertExecutor


class EPBalanceStrategy:
    """Own the duplicated-expert plan and executor for one expert module."""

    def __init__(self, ep_group: dist.ProcessGroup, num_experts: int, max_dup_experts_num: int) -> None:
        """Initialize balancing for the EP group and its global expert count."""
        # Keep the optional Numba planner out of ordinary, unbalanced EP imports.
        from .greedy_dup_experts_planner import GreedyDupExpertsPlanner

        if num_experts <= 0 or num_experts % dist.get_world_size(ep_group):
            raise ValueError("The number of experts must be positive and divisible by EP size.")
        if max_dup_experts_num < 0:
            raise ValueError("max_dup_experts_num must be non-negative.")
        self.ep_group = ep_group
        self.num_experts = num_experts
        self.planner: GreedyDupExpertsPlanner = GreedyDupExpertsPlanner(
            ep_group=ep_group, num_experts=num_experts, max_dup_experts_num=max_dup_experts_num
        )
        self.executor = DupExpertExecutor(
            ep_group=ep_group, num_experts=num_experts, max_dup_experts_num=max_dup_experts_num
        )

    def _plan(self, top_k_index: torch.Tensor) -> None:
        if TrainingContext().get_training_stage() == TrainingStage.BACKWARD:
            return  # Recompute must reuse the forward plan.
        num_local_tokens_per_expert = get_num_global_tokens_per_local_expert(top_k_index, self.num_experts)
        ep_size = dist.get_world_size(self.ep_group)
        if ep_size == 1:
            num_global_tokens_per_expert = num_local_tokens_per_expert.view(1, -1)
        else:
            num_global_tokens_per_expert = torch.zeros(
                ep_size,
                self.num_experts,
                dtype=num_local_tokens_per_expert.dtype,
                device=num_local_tokens_per_expert.device,
            )
            dist.all_gather_into_tensor(
                num_global_tokens_per_expert, num_local_tokens_per_expert.view(1, -1), group=self.ep_group
            )
        with torch.no_grad():
            self.planner.select_dup_experts_and_rearrange_experts(num_global_tokens_per_expert, top_k_index)

    def prepare_all2all(
        self, top_k_index: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Return balanced routes, permutation metadata and AllToAll splits.

        Forward builds a plan; backward recomputation reads the cached plan.
        """
        self._plan(top_k_index)
        num_local_experts = self.planner.num_local_experts + self.planner.max_dup_experts_num
        expert_ids = torch.arange(
            num_local_experts * dist.get_world_size(self.ep_group), dtype=torch.int32, device=top_k_index.device
        ).remainder(num_local_experts)
        global_indices = torch.repeat_interleave(
            expert_ids, self.planner.get_plan_item("num_global_tokens_per_local_expert").ravel()
        )
        return (
            self.planner.get_plan_item("selected_experts_with_dup"),
            (self.planner.get_plan_item("num_global_sum_tokens_per_local_expert"), global_indices),
            (self.planner.get_plan_item("input_splits"), self.planner.get_plan_item("output_splits")),
        )

    def prepare_allgather(self, top_k_index: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Return gathered balanced routes, local expert count and token counts.

        The caller sets EP sequence lengths before this call, as for ordinary
        AllGather. Recompute reuses the gathered routes from the forward plan.
        """
        if TrainingContext().get_training_stage() == TrainingStage.FORWARD:
            self._plan(top_k_index)
            with torch.no_grad():
                self.planner.plan_cache[-1]["selected_experts_with_dup"] = allgather_tokens_in_ep(
                    self.planner.get_plan_item("selected_experts_with_dup"), self.ep_group
                )
        return (
            self.planner.get_plan_item("selected_experts_with_dup"),
            self.planner.num_local_experts + self.planner.max_dup_experts_num,
        )


def initialize_ep_balance(module: torch.nn.Module, ep_group: dist.ProcessGroup, max_dup_experts_num: int) -> None:
    """Attach a strategy to an EP expert module when balancing is enabled.

    Preserve an existing strategy; the expert module already has its global
    expert count assigned by EP initialization.
    """
    if getattr(module, "ep_balance_strategy", None) is None:
        module.ep_balance_strategy = EPBalanceStrategy(ep_group, module.num_global_experts, max_dup_experts_num)
