import os
import logging

import torch

from fsdp_turbo.utils.log import print_rank

logger = logging.getLogger(__name__)


class Checkpointer:
    """Basic checkpoint manager that saves/loads model, optimizer, and scheduler states."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def save(self, model, optimizer, scheduler, global_step):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        state = {"global_step": global_step}
        if model is not None:
            state["model_state_dict"] = model.state_dict()
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        ckpt_path = os.path.join(ckpt_dir, "training_state.pt")
        torch.save(state, ckpt_path)
        print_rank(logger.info, f"Checkpoint saved to {ckpt_path}")

    def load(self, checkpoint_path, model, optimizer, scheduler):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if model is not None and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        print_rank(logger.info, f"Checkpoint loaded from {checkpoint_path}")
        return state.get("global_step", 0)
