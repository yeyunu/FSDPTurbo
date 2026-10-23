from typing import Callable, List, Optional

from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from fsdp_turbo.distributed.expert_parallel.utils import get_num_global_tokens_per_local_expert
from fsdp_turbo.distributed.expert_parallel.dispatcher import experts_computation, apply_activation
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig
from fsdp_turbo.ops.moe import grouped_matmul, permute, unpermute


def _group_size(group):
    # Follow c10d convention: None denotes the default process group.
    return dist.get_world_size(group) if dist.is_initialized() else 1


def _a2a_single_async(
    inputs: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    output_split_sizes: List = None,
    input_split_sizes: List = None,
    async_op: bool = True,
):
    if group is None or dist.get_world_size(group=group) == 1:
        return None, inputs
    assert inputs.shape[0] == sum(input_split_sizes), (
        f"inputs.shape[0]={inputs.shape[0]} != sum(input_split_sizes)={sum(input_split_sizes)}; "
        f"input_split_sizes={input_split_sizes}"
    )
    inputs = inputs.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(inputs)
    else:
        output = inputs.new_empty(
            size=[sum(output_split_sizes)] + list(inputs.size()[1:]), dtype=inputs.dtype, device=inputs.device
        )
    if async_op:
        work = dist.all_to_all_single(
            output,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    else:
        dist.all_to_all_single(
            output,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        work = None
    return work, output


class _A2AStart(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, group, output_split_sizes, input_split_sizes):
        work, output = _a2a_single_async(inputs, group, output_split_sizes, input_split_sizes, async_op=True)
        handle = [work]
        ctx.comm = handle
        return handle, output

    @staticmethod
    def backward(ctx, grad_comm, grad_output):
        if ctx.comm:
            work = ctx.comm.pop()
            if work is not None:
                work.wait()
        return None, None, None, None


class _A2AEnd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, comm, inputs, output, group, output_split_sizes, input_split_sizes):
        ctx.comm = comm
        if ctx.comm:
            work = ctx.comm.pop()
            if work is not None:
                work.wait()
        ctx.group = group
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        return output

    @staticmethod
    def backward(ctx, grad_y):
        work, grad_input = _a2a_single_async(
            grad_y.contiguous(),
            ctx.group,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            async_op=True,
        )
        ctx.comm.append(work)
        return None, grad_input, grad_y, None, None, None


@dataclass
class _ChunkMetadata:
    input_splits: list[int]
    output_splits: list[int]
    tokens_per_rank: torch.Tensor
    tokens_per_expert: torch.Tensor


def batched_pre_dispatch_metadata(chunk_indices, num_global_experts, ep_group=None):
    """One all-gather and one blocking host transfer for all chunk splits."""
    ep_size = _group_size(ep_group)
    ep_rank = dist.get_rank(ep_group) if ep_size > 1 else 0
    if num_global_experts <= 0 or num_global_experts % ep_size:
        raise ValueError("The number of experts must be positive and divisible by EP size.")
    num_local_experts = num_global_experts // ep_size
    counts = torch.stack(
        [get_num_global_tokens_per_local_expert(indices, num_global_experts) for indices in chunk_indices]
    )
    if ep_size > 1:
        gathered = counts.new_empty((ep_size, *counts.shape))
        dist.all_gather_into_tensor(gathered.view(-1), counts.view(-1), group=ep_group)
    else:
        gathered = counts.unsqueeze(0)
    # [ranks, chunks, experts] -> [chunks, ranks, local_experts]
    local_counts = gathered[:, :, ep_rank * num_local_experts : (ep_rank + 1) * num_local_experts]
    local_counts = local_counts.transpose(0, 1).contiguous()
    sends = counts.reshape(len(chunk_indices), ep_size, num_local_experts).sum(-1)
    receives = local_counts.sum(-1)
    host_splits = torch.stack((sends, receives), dim=1).cpu().tolist()
    return [
        _ChunkMetadata(splits[0], splits[1], per_rank, per_rank.sum(0))
        for splits, per_rank in zip(host_splits, local_counts)
    ]


def post_dispatch_pre_combine(
    hidden_states: torch.Tensor,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    num_global_sum_tokens_per_local_expert: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
    activation: Optional[Callable] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
):
    post_dispatch_unpermute_indices = None
    if hidden_states.shape[0] > 0:
        ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        num_local_experts = num_global_experts // ep_size
        if num_global_experts % ep_size != 0:
            raise ValueError(
                f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
            )
        _expert_ids_per_ep_rank = (
            torch.arange(num_global_experts, dtype=torch.int32, device=hidden_states.device) % num_local_experts
        )
        global_input_tokens_local_experts_indices = torch.repeat_interleave(
            _expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
        )
        hidden_states, post_dispatch_unpermute_indices = permute(
            hidden_states, global_input_tokens_local_experts_indices, use_eager=not fused
        )
        intermediate_hidden_states = grouped_matmul(
            hidden_states,
            fc1_weight,
            num_global_sum_tokens_per_local_expert,
            use_eager=not fused,
        )
        intermediate_activations = apply_activation(
            intermediate_hidden_states,
            dim=-1,
            fused=fused,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            activation=activation,
        )
        hidden_states = grouped_matmul(
            intermediate_activations, fc2_weight, num_global_sum_tokens_per_local_expert, use_eager=not fused
        )
        hidden_states = unpermute(hidden_states, post_dispatch_unpermute_indices, use_eager=not fused)
    else:
        intermediate_hidden_states = hidden_states @ fc1_weight.sum(0)
        gate_output, down_output = torch.chunk(intermediate_hidden_states, 2, dim=-1)
        hidden_states = (gate_output + down_output) @ fc2_weight.sum(0) * 0.0
    return hidden_states


def chunkmoe_experts_forward_fn(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    weights: torch.Tensor,
    num_global_experts: int,
    expert_ids_per_ep_rank: list[int],
    ep_group: torch.distributed.ProcessGroup,
    fused: bool,
    act_fn: Optional[Callable] = None,
    act_limit: Optional[float] = None,
    act_alpha: Optional[float] = None,
    ep_balance_strategy=None,
    quant_config=None,
    expert_module=None,
    ep_plan: EPPlanConfig = None,
    **kwargs,
):
    """Dispatch contiguous token chunks in pairs, padding rank tails empty.

    Chunk count must agree across ranks even when their token counts differ.
    All-empty batches still execute one chunk to preserve the autograd graph.
    """

    num_tokens = hidden_states.shape[0]
    if ep_plan.moe_num_chunks:
        num_chunks = ep_plan.moe_num_chunks
        chunk_size = max(1, (num_tokens + num_chunks - 1) // num_chunks)
    else:
        chunk_size = ep_plan.moe_chunk_size or max(1, num_tokens)
        num_chunks = max(1, (num_tokens + chunk_size - 1) // chunk_size)
        if _group_size(ep_group) > 1:
            count = torch.tensor(num_chunks, device=hidden_states.device, dtype=torch.int64)
            dist.all_reduce(count, op=dist.ReduceOp.MAX, group=ep_group)
            num_chunks = int(count.item())

    def split(tensor):
        return [
            tensor[min(c * chunk_size, num_tokens) : min((c + 1) * chunk_size, num_tokens)] for c in range(num_chunks)
        ]

    chunks_hs, chunks_index, chunks_weights = map(split, (hidden_states, top_k_index, top_k_weights))
    metadata = batched_pre_dispatch_metadata(chunks_index, num_global_experts, ep_group)

    def forward_pair(*args):
        # Tensor arguments are explicit checkpoint inputs. The final argument
        # identifies the pair; no loop variable is captured by reference.
        *tensors, first = args
        pair_metadata = metadata[first : first + len(tensors) // 3]
        dispatched, local_inputs, inverse_indices, modes = [], [], [], []
        for offset, meta in enumerate(pair_metadata):
            hs, indices, _ = tensors[3 * offset : 3 * offset + 3]
            # Empty fused permutations are unsupported on some devices.
            mode = fused and hs.shape[0] > 0 and sum(meta.output_splits) > 0
            local, inverse = permute(hs, indices.to(torch.int32), use_eager=not mode)
            pending, received = _A2AStart.apply(local, ep_group, meta.output_splits, meta.input_splits)
            dispatched.append((pending, received))
            local_inputs.append(local)
            inverse_indices.append(inverse)
            modes.append(mode)

        combined = []
        for offset, meta in enumerate(pair_metadata):
            pending, received = dispatched[offset]
            received = _A2AEnd.apply(
                pending,
                local_inputs[offset],
                received,
                ep_group,
                meta.output_splits,
                meta.input_splits,
            )
            local_ids = torch.repeat_interleave(expert_ids_per_ep_rank, meta.tokens_per_rank.reshape(-1))
            local, inverse = permute(received, local_ids, use_eager=not modes[offset])
            local = experts_computation(
                local,
                meta.tokens_per_expert,
                *weights,
                modes[offset],
                act_fn=act_fn,
                act_limit=act_limit,
                act_alpha=act_alpha,
                quant_config=quant_config,
                weight_owner=expert_module,
            )
            local = unpermute(local, inverse, use_eager=not modes[offset])
            pending, returned = _A2AStart.apply(local, ep_group, meta.input_splits, meta.output_splits)
            combined.append((pending, local, returned))

        outputs = []
        for offset, meta in enumerate(pair_metadata):
            pending, local, returned = combined[offset]
            returned = _A2AEnd.apply(
                pending,
                local,
                returned,
                ep_group,
                meta.input_splits,
                meta.output_splits,
            )
            outputs.append(
                unpermute(
                    returned,
                    inverse_indices[offset],
                    tensors[3 * offset + 2],
                    use_eager=not modes[offset],
                )
            )
        return tuple(outputs)

    outputs = []
    for first in range(0, num_chunks, 2):
        args = [
            tensor
            for c in range(first, min(first + 2, num_chunks))
            for tensor in (chunks_hs[c], chunks_index[c], chunks_weights[c])
        ]
        if ep_plan.chunk_moe_recompute:
            # Empty chunks can save fewer tensors on one rank. Full replay is
            # required so all ranks issue the same collectives during backward.
            with set_checkpoint_early_stop(False):
                outputs.extend(checkpoint(forward_pair, *args, first, use_reentrant=False))
        else:
            outputs.extend(forward_pair(*args, first))
    result = torch.cat(outputs, dim=0)
    return result
