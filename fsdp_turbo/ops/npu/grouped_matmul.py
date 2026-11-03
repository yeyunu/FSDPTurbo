import torch
from fsdp_turbo.ops.registry import register_op

try:
    import torch_npu
except ImportError:
    torch_npu = None


class GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, weights, weights_bias, m_split, group_list_type, weights_layout):
        # Due to ascend gmm kernel k split limitations, we need a tensor m_split, not a tensor List.
        if not isinstance(m_split, torch.Tensor):
            ctx.group_list = torch.tensor(m_split, device='npu', dtype=torch.int64)
        else:
            ctx.group_list = m_split

        ctx.group_list_type = group_list_type
        ctx.weights_layout = weights_layout
        ctx.weight_shape = weights.shape
        if weights.dim() == 4:
            weights = weights.reshape(weights.shape[0], weights.shape[2], weights.shape[3])
        if weights.dim() != 3:
            raise ValueError(f"grouped_matmul expects 3-D/4-D expert weights, got shape {tuple(weights.shape)}")
        # The op consumes one stacked input-major [G, K, N] weight. The linear
        # convention is transposed as a view; the layout is explicit because
        # square weights cannot be told apart by shape.
        weights_kn = weights if weights_layout == "input_major" else weights.transpose(1, 2)
        ctx.save_for_backward(input_tensor, weights)

        fwd_output = torch_npu.npu_grouped_matmul(
            [input_tensor],
            [weights_kn],
            bias=weights_bias,
            group_list=ctx.group_list,
            split_item=2,
            group_type=0,
            group_list_type=ctx.group_list_type,
        )[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        group_list = ctx.group_list
        inp, weights = ctx.saved_tensors
        group_list_type = ctx.group_list_type

        # dX = grad_output @ W: the op wants W in [G, N, K] layout. Linear
        # weights already are [G, N, K]; input-major weights need a view.
        dx_weight = weights.transpose(1, 2) if ctx.weights_layout == "input_major" else weights
        grad = torch_npu.npu_grouped_matmul(
            [grad_output],
            [dx_weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
        )[0]

        # dW = inp^T @ grad_output is produced input-major [G, K, N]; transpose
        # it back when the caller passed linear-convention weights.
        grad_weight = torch_npu.npu_grouped_matmul(
            [inp.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=group_list_type,
        )[0]
        if ctx.weights_layout == "linear":
            grad_weight = grad_weight.transpose(1, 2)
        if grad_weight.shape != ctx.weight_shape:
            grad_weight = grad_weight.reshape(ctx.weight_shape)
        return grad, grad_weight, None, None, None, None


@register_op('grouped_matmul', 'npu')
def grouped_matmul_npu(inputs, m_split, weights, weights_layout="linear"):
    return GroupedMatmul.apply(inputs, weights, None, m_split, 1, weights_layout)


@register_op('w4a16_grouped_matmul', 'npu')
def w4a16_grouped_matmul_npu(
    x,
    weights,
    bias=None,
    group_list=None,
    group_type=0,
    output_dtype=None,
    group_list_type=0,
    split_item=2,
):
    """Low-precision (W4A16) grouped matmul on Ascend.

    Wraps ``torch_npu.npu_grouped_matmul`` with the W4A16 QAT recipe's
    argument set (``output_dtype``, ``group_type``, ``split_item``).
    """
    return torch_npu.npu_grouped_matmul(
        [x],
        weights,
        bias=bias,
        group_list=group_list,
        group_type=group_type,
        output_dtype=output_dtype,
        group_list_type=group_list_type,
        split_item=split_item,
    )
