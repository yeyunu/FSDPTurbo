"""Device-dispatched gated delta rule with explicit backend selection."""

import torch

from fsdp_turbo.ops.registry import dispatch_op


def chunk_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    head_first: bool = False,
    implementation: str = 'auto',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute GDN in BTHD layout, preserving state and packed sequence boundaries.

    ``eager`` selects the portable implementation on the input device.
    ``ascendc`` requires the NPU kernel and never silently falls back.
    No checkpoint replay or training-context arguments are accepted. AscendC
    supports BF16 initial-state gradients on Ascend 950; final-state gradients require eager.
    """
    device_type = 'cpu' if implementation == 'eager' else q.device.type
    if implementation == 'ascendc' and device_type != 'npu':
        raise ValueError('AscendC GDN requires NPU tensors.')
    backend_options = {'implementation': implementation} if device_type == 'npu' else {}
    return dispatch_op(
        'chunk_gated_delta_rule',
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        head_first=head_first,
        device_type=device_type,
        **backend_options,
    )


def fused_recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    implementation: str = 'auto',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute recurrent GDN for cached continuation without mutating the input state."""
    device_type = 'cpu' if implementation == 'eager' else q.device.type
    return dispatch_op(
        'fused_recurrent_gated_delta_rule',
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        device_type=device_type,
    )


__all__ = [
    "chunk_gdn",
    "fused_recurrent_gdn",
]
