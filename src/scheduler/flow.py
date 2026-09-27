from typing import Any, Callable, Optional, Tuple

import torch
from torch.distributions import Beta, LogisticNormal


class RectifiedFlow:
    def __init__(self, sampling: str = "beta"):
        self.sampling = sampling

        self.logistic_normal = LogisticNormal(0.0, 1.0)
        self.beta = Beta(1.5, 1.0)

    @torch.no_grad
    def generate(
        self,
        x0: torch.Tensor,
        num_steps: int = 5,
        forward_func: Optional[Callable[[torch.Tensor, torch.Tensor, Any], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate sample x1 (target) from noise x0 using Euler integration.

        Args:
            x0: initial noise tensor, shape (batch_size, input_dim)
            num_steps: number of integration steps
            forward_func: velocity field prediction function, takes (z_t, t, **kwargs)
            **kwargs: additional arguments for forward_func

        Returns:
            x1: final generated sample (target data)
        """
        dt = 1.0 / num_steps
        z = x0.clone()  # Start from noise

        for step in range(num_steps):
            t = torch.ones((z.shape[0], 1, 1), device=z.device, dtype=z.dtype) * step / num_steps
            v = forward_func(z, t, **kwargs)
            z = z + v * dt

        return z  # z now approximates x1 (target)

    @torch.no_grad
    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation between x0 (noise) and x1 (target): z_t = (1 - t) * x0 + t * x1

        Args:
            x0: initial noise tensor
            x1: clean data tensor
            t: time step (scalar or batched)

        Returns:
            z_t: interpolated tensor at time t
        """
        return (1 - t) * x0 + t * x1

    @torch.no_grad
    def get_velocity_target(self, x0: torch.Tensor, x1: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the target velocity field for training.

        Args:
            x0: initial noise tensor
            x1: clean data tensor
            t: time step (not used in linear case)

        Returns:
            velocity: target velocity field v = x1 - x0
        """
        return x1 - x0

    @torch.no_grad
    def get_timestep(self, batch_size: int, dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> torch.Tensor:
        if self.sampling == "logit_normal":
            u = self.logistic_normal.sample((batch_size,))[:, 0].to(device)
        elif self.sampling == "beta":
            u = self.beta.sample((batch_size,)).to(device)
            u = (1 - u) * 0.999
        else:
            u = torch.rand(size=(batch_size,), device=device)
        return u.to(dtype)

    @torch.no_grad
    def training_data(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training data for Rectified Flow.

        Args:
            x0: noise tensor, shape (batch_size, input_dim)
            x1: clean data tensor, shape (batch_size, input_dim)

        Returns:
            z_t: interpolated sample
            target_velocity: velocity field target
            t: random time step
        """
        batch_size = x1.shape[0]
        device = x1.device
        dtype = x1.dtype

        t = self.get_timestep(batch_size, dtype, device).unsqueeze(dim=1).unsqueeze(dim=1)
        z_t = self.interpolate(x0, x1, t)
        target_velocity = self.get_velocity_target(x0, x1, t)

        return z_t, target_velocity, t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scheduler = RectifiedFlow()
    u = scheduler.get_timestep(100000)
    plt.hist(u.float().cpu().numpy(), bins=100)
    plt.savefig("hist.png")
