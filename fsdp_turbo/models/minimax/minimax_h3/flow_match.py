"""H3 flow-matching schedule, derived from pinned DiffSynth (Apache-2.0)."""

import torch


class FlowMatchScheduler:
    """Pinned H3 sigma schedule for video/audio Flow Matching training."""

    def __init__(self, template: str = "MiniMax-H3") -> None:
        """Initialize the flow-matching schedule with CPU timestep lookup."""
        self.set_timesteps_fn = self.set_timesteps_minimax_h3
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_minimax_h3(
        num_inference_steps: int = 50, denoising_strength: float = 1.0, shift: float = 12.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build shifted sigmas and corresponding diffusion timesteps."""
        num_train_timesteps = 1000
        base = torch.linspace(denoising_strength, 0.0, num_inference_steps + 1, dtype=torch.float32)[:-1]
        sigmas = shift * base / (1 + (shift - 1) * base)
        timesteps = sigmas * num_train_timesteps
        return (sigmas, timesteps)

    def set_training_weight(self) -> None:
        """Normalize timestep-dependent weights for the training schedule."""
        steps = 1000
        x = self.sigmas * self.num_train_timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def set_timesteps(
        self, num_inference_steps: int = 100, denoising_strength: float = 1.0, training: bool = False, **kwargs: float
    ) -> None:
        """Set diffusion timesteps and optional training weights."""
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps, denoising_strength=denoising_strength, **kwargs
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Interpolate between clean samples and noise at a diffusion timestep."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Return the flow-matching target velocity."""
        target = noise - sample
        return target

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Look up the loss weight for a diffusion timestep."""
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
