from typing import Callable, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate


class ChunkLoss(torch.autograd.Function):
    """
    A custom autograd function that computes a loss in chunks to reduce memory usage.

    This function splits the input hidden states along the feature dimension into chunks,
    computes the loss and gradients for each chunk separately using a provided loss function,
    and then accumulates the results. It is particularly useful when the full forward pass
    would exceed device memory limits.

    Note: Bias terms in the head (e.g., classifier bias) are not currently supported.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: Optional[torch.Tensor],
        loss_forward: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        loss_kwargs_chunks: list[dict[str, object]],
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Compute chunk gradients, accumulating standard linear-head dW in place.

        The callback uses F.linear with the supplied dtypes and current autocast,
        returning loss and FP32 logits;
        the loss must depend on hidden states and weights only through those logits.
        dW uses the common weight/autocast dtype to
        avoid narrowing BF16 projection operands to FP16 before the GEMM.
        Rounding can differ from separate per-chunk matmul and add operations.

        Args:
            ctx: Context object used to save tensors for backward pass.
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            head_weight (torch.Tensor): Weight matrix of the final prediction head.
            head_bias (Optional[torch.Tensor]): Bias vector of the head (currently unsupported).
            loss_forward: Linear-head loss callback returning (loss, logits).
            loss_kwargs_chunks: A list of keyword argument dictionaries, one per chunk.
            chunk_size (int): The size (in feature dimension) of each chunk.

        Returns:
            torch.Tensor: Scalar tensor representing the accumulated loss over all chunks.
        """

        if head_bias is not None:
            raise NotImplementedError("head_bias is not supported in ChunkLoss")

        # Handle FSDP-sharded DTensor head_weight: torch.func.grad_and_value
        # does not preserve the DTensor type, which would leave head_weight.grad
        # as a plain Tensor while the optimizer state is a DTensor, causing
        # "got mixed torch.Tensor and DTensor" errors in foreach operations.
        is_dtensor = isinstance(head_weight, DTensor)
        if is_dtensor:
            device_mesh = head_weight.device_mesh
            placements = head_weight.placements
            # All-gather to full weight so loss is computed on the complete
            # vocabulary (correct cross-entropy) and torch.func works on a
            # plain Tensor without DTensor dispatch issues.
            head_weight = head_weight.full_tensor()

        device = hidden_states.device
        # Initialize accumulated scalar loss on the same device
        accumulated_loss = torch.tensor(0.0, device=device)
        # Pre-allocate gradient tensors for inputs and weights
        grad_inputs = torch.empty_like(hidden_states)
        projection_dtype = head_weight.dtype
        grad_dtype = head_weight.dtype
        if torch.is_autocast_enabled(device.type):
            if projection_dtype != torch.float64:
                projection_dtype = torch.get_autocast_dtype(device.type)
            grad_dtype = torch.promote_types(grad_dtype, projection_dtype)
        grad_weight = torch.zeros_like(head_weight, dtype=grad_dtype)

        # Split tensors into chunks along the feature dimension (dim=1 assumed to be sequence dim)
        grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
        hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)

        # Process each chunk independently
        for hidden_states_chunk, grad_inputs_chunk, loss_kwargs in zip(
            hidden_states_chunks, grad_inputs_chunks, loss_kwargs_chunks
        ):
            flat_hidden = hidden_states_chunk.reshape(-1, head_weight.size(1))
            with torch.enable_grad():
                per_chunk_loss, logits = loss_forward(
                    hidden_states_chunk.detach().requires_grad_(), head_weight, None, **loss_kwargs
                )
                (grad_logits,) = torch.autograd.grad(per_chunk_loss, logits)
            grad_logits = grad_logits.reshape(-1, head_weight.size(0))
            grad_logits = grad_logits.to(projection_dtype)
            chunk_grad_input = grad_logits.matmul(head_weight.to(projection_dtype)).reshape_as(hidden_states_chunk)
            # addmm_ does not autocast: use the rounded projection operands,
            # then accumulate directly into dW without a per-chunk [vocab, hidden] gradient.
            grad_weight.addmm_(
                grad_logits.to(grad_weight.dtype).t(), flat_hidden.to(projection_dtype).to(grad_weight.dtype)
            )
            del logits, grad_logits, flat_hidden

            # Accumulate loss and gradients
            accumulated_loss.add_(per_chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            del chunk_grad_input, per_chunk_loss

        grad_weight = grad_weight.to(head_weight.dtype)
        # Convert gradient back to DTensor with the original placement so that
        # head_weight.grad matches the parameter type expected by the optimizer.
        if is_dtensor:
            grad_weight = DTensor.from_local(grad_weight, device_mesh, [Replicate()] * len(placements))
            grad_weight = grad_weight.redistribute(device_mesh, placements)

        # Save computed gradients for use in backward pass
        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss

    @staticmethod
    def backward(ctx, *grad_output) -> Tuple:
        """
        Backward pass: propagate upstream gradients through the precomputed gradients.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss w.r.t. the output (usually a scalar).

        Returns:
            tuple: Gradients w.r.t. (hidden_states, head_weight, head_bias, loss_forward,
                   loss_kwargs_chunks, chunk_size). Only the first two are non-None.
        """
        grad_input, grad_weight = ctx.saved_tensors
        # Always apply the upstream scalar.  Comparing a device scalar in a
        # Python ``if`` forces an NPU synchronization and can surface an
        # earlier asynchronous HCCL error as an unrelated stream failure.
        grad_input = grad_input * grad_output[0]
        grad_weight = grad_weight * grad_output[0]

        return grad_input, grad_weight, None, None, None, None


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "sum",
    **kwargs,
) -> torch.Tensor:
    """
    Compute a modified cross-entropy loss that optionally normalizes the loss by a per-example or global scaling factor (alpha).
    Args:
        source (torch.Tensor): Predicted logits of shape (N, C), where C is the number of classes.
        target (torch.Tensor): Ground truth labels of shape (N,) with values in [0, C-1].
        alpha (Optional[torch.Tensor]): Optional scaling factor.
            - If scalar (0-D tensor), it globally scales the total loss.
            - If 1-D tensor of shape (N,), it provides per-example scaling factors.
            - If 2-D tensor of shape (B, S), it provides per-token scaling factors.
            - Must be positive and non-zero.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the loss.
        reduction (str): Specifies the reduction to apply to the output:
            - `"sum"`: Sum all losses before dividing by `alpha`.
            - `"none"`: No reduction; used here to enable per-example weighting via `alpha`.
            Note: `"mean"` is not supported in this implementation.
        **kwargs: Additional keyword arguments passed to `F.cross_entropy` (though currently unused).

    Returns:
        torch.Tensor: A scalar tensor representing the normalized cross-entropy loss.
    """

    # Compute standard cross-entropy loss
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if alpha is not None:
        alpha = alpha.to(loss.device)

    if reduction == "sum":
        # Global normalization: total loss divided by scalar alpha
        loss = loss / alpha

    elif reduction == "none":
        if alpha.ndim == 0:
            # Alpha is a scalar: sum all element-wise losses, then divide
            loss = loss.sum() / alpha
        elif alpha.ndim >= 2 and alpha.shape[1] > 1:
            # Normalize each token separately, including packed samples in one row.
            loss = loss.view(alpha.shape[0], -1)
            loss = (loss / alpha).sum()
        else:
            # Alpha is 1-D with shape (N,): assume N examples
            # Reshape loss to (N, -1) to group elements per example
            loss = loss.view(alpha.shape[0], -1)
            # Sum over non-batch dimensions (e.g., sequence length in token classification)
            # Normalize each example's loss by its corresponding alpha
            loss = loss.sum(1) / alpha
            loss = loss.sum()
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}. Use 'sum' or 'none'.")

    return loss


def calculate_lm_loss(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: Optional[torch.Tensor] = None,
    *,
    shift_labels: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the language modeling (LM) loss using a linear output head and a modified cross-entropy function.

    This function is typically used in autoregressive language models where the prediction for each token
    is compared against the next token in the sequence. The input `hidden_states` are projected to logits
    via a linear layer (without bias if `head_bias` is None), then compared to shifted labels.

    Note:
        - The `head_bias` argument is accepted for interface compatibility but **not used** in the current implementation.
        - Label shifting (i.e., aligning predictions with next-token targets) is assumed to have been done externally;
          this function only flattens the tensors for loss computation.

    Args:
        hidden_states (torch.Tensor):
            The hidden representations from the transformer backbone, of shape (batch_size, seq_len, hidden_dim).
        head_weight (torch.Tensor):
            Weight matrix of the output classification head, of shape (vocab_size, hidden_dim).
        head_bias (Optional[torch.Tensor], optional):
            Bias vector for the output head (shape: (vocab_size,)). Currently **ignored**.
        shift_labels (torch.Tensor):
            Ground truth token IDs, already shifted to align with predictions (e.g., target[i] = input[i+1]),
            of shape (batch_size, seq_len).
        alpha (Optional[torch.Tensor], optional):
            Optional scaling factor used in `fixed_cross_entropy` for loss normalization
            (e.g., per-example or global weighting). See `fixed_cross_entropy` for details.
        ignore_index (int, optional):
            Specifies a target value that is ignored and does not contribute to the loss. Default: -100.
        reduction (Optional[str], optional):
            Reduction method for the loss ('none', 'sum', etc.). Passed directly to `fixed_cross_entropy`.
            If None, the default behavior of `fixed_cross_entropy` applies (typically 'sum').
        **kwargs (Any):
            Additional keyword arguments forwarded to `fixed_cross_entropy`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - **loss**: Scalar tensor representing the computed LM loss.
            - **logits**: The unnormalized prediction scores of shape (batch_size * seq_len, vocab_size).

    Example:
        >>> hidden = torch.randn(2, 5, 768)
        >>> weight = torch.randn(30522, 768)
        >>> labels = torch.randint(0, 30522, (2, 5))
        >>> loss, logits = calculate_lm_loss(hidden, weight, shift_labels=labels, reduction="sum")
    """
    # Flatten labels to 1D: (batch_size * seq_len,)
    shift_labels = shift_labels.reshape(-1)

    # Flatten hidden states to (batch_size * seq_len, hidden_dim)
    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))

    # Project to logits using only the weight (bias is intentionally omitted here)
    # Cast to float to ensure numerical stability in loss computation
    logits = F.linear(hidden_states, head_weight).float()

    # Compute the modified cross-entropy loss
    loss = fixed_cross_entropy(
        logits, shift_labels, alpha=alpha, ignore_index=ignore_index, reduction=reduction, **kwargs
    )

    return loss, logits


def chunk_loss(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: Optional[torch.Tensor],
    loss_forward: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    loss_kwargs_chunks: list[dict[str, object]],
    chunk_size: int,
) -> torch.Tensor:
    """
    Compute loss in chunks using the custom autograd function `ChunkLoss`.

    Merge batch and sequence into one token axis, keeping labels and alpha
    aligned. Callers can pass one dictionary with unsplit labels/alpha, or
    existing sequence-split dictionaries. Other loss kwargs must be shared.
    Alpha is a scalar, a per-sample [B] denominator, or per-token [B, S].
    Label shifting and normalization denominators are owned by the caller.

    For sum reduction with a scalar alpha, filter ignored tokens before the
    output projection and re-chunk valid tokens across the batch. Boolean
    indexing scatters input gradients back, leaving ignored rows at zero.
    Other reductions and fully valid or fully ignored batches use the dense path.
    Weight gradients accumulate in place, promoting mixed FP16/BF16 to FP32,
    and return in the stored weight dtype.

    Args:
        hidden_states: Input tensor (e.g., from a transformer) to compute loss on.
        head_weight: Weight matrix of the output classification head.
        head_bias: Bias vector of the head.
        loss_forward: Callback returning loss and logits from a standard linear head,
            with loss depending on hidden states and weights only through logits.
        loss_kwargs_chunks: Aligned labels/alpha and shared kwargs for `loss_forward`.
        chunk_size: Maximum total tokens per chunk, independent of batch size.
            Any positive integer is supported; the final chunk can be shorter.

    Returns:
        The total accumulated loss as a scalar tensor.
    """
    loss_kwargs = loss_kwargs_chunks[0]
    shift_labels = torch.cat([chunk["shift_labels"] for chunk in loss_kwargs_chunks], dim=1)
    alpha = loss_kwargs.get("alpha")
    if torch.is_tensor(alpha) and alpha.ndim > 0:
        alpha_chunks = []
        for chunk in loss_kwargs_chunks:
            chunk_alpha = chunk["alpha"]
            if chunk_alpha.ndim == 1:
                chunk_alpha = chunk_alpha.unsqueeze(1)
            alpha_chunks.append(chunk_alpha.expand_as(chunk["shift_labels"]))
        alpha = torch.cat(alpha_chunks, dim=1).reshape(1, -1)

    hidden_states = hidden_states.reshape(1, -1, hidden_states.shape[-1])
    shift_labels = shift_labels.reshape(1, -1)
    if loss_kwargs.get("reduction") == "sum" and torch.is_tensor(alpha) and alpha.ndim == 0:
        ignore_index = loss_kwargs.get("ignore_index", -100)
        valid_tokens = shift_labels.reshape(-1) != ignore_index
        valid_token_count = int(valid_tokens.sum())

        if 0 < valid_token_count < valid_tokens.numel():
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))[valid_tokens].unsqueeze(0)
            shift_labels = shift_labels.reshape(-1)[valid_tokens].unsqueeze(0)

    if torch.is_tensor(alpha) and alpha.ndim > 0:
        alpha = torch.split(alpha, chunk_size, dim=1)
    loss_kwargs_chunks = [
        {**loss_kwargs, "shift_labels": labels, "alpha": alpha[index] if isinstance(alpha, tuple) else alpha}
        for index, labels in enumerate(torch.split(shift_labels, chunk_size, dim=1))
    ]

    return ChunkLoss.apply(hidden_states, head_weight, head_bias, loss_forward, loss_kwargs_chunks, chunk_size)


def chunk_lm_loss(hidden_states, head_weight, loss_kwargs_chunks, chunk_size):
    """Chunked LM loss kernel for callers that own the head projection.

    Used as the injectable compute step of the CP loss helper: the caller
    obtains ``head_weight`` from inside the FSDP-managed ``lm_head`` forward
    and passes the pre-built per-chunk loss kwargs here.
    """
    return chunk_loss(hidden_states, head_weight, None, calculate_lm_loss, loss_kwargs_chunks, chunk_size)
