import logging

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


class Monitor:
    """Basic training monitor that tracks loss history and logs metrics."""

    def __init__(self, logging_steps: int = 10):
        self.logging_steps = logging_steps
        self.loss_history = []

    def log(self, step: int, loss: float):
        self.loss_history.append((step, loss))

    def summary(self):
        if not self.loss_history:
            return
        losses = [l for _, l in self.loss_history]
        avg_loss = sum(losses) / len(losses)
        print_rank(logger.info, f"Training summary: {len(self.loss_history)} steps, avg loss = {avg_loss:.6f}")
