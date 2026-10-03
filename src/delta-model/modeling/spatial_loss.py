"""Spatial distribution losses for source-contribution rasters."""

import torch
from torch import nn
from torch.nn import functional as F


class UnbalancedSinkhornLoss(nn.Module):
    """Compare nonnegative rasters with a debiased unbalanced transport loss."""

    def __init__(
        self,
        grid_size: int,
        *,
        blur_cells: float,
        reach_cells: float,
        iterations: int,
    ) -> None:
        super().__init__()
        offsets = torch.arange(-(grid_size - 1), grid_size, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (offsets / blur_cells).square())
        self.register_buffer("kernel", kernel)
        self.entropy = (blur_cells / reach_cells) ** 2
        self.mass_exponent = 1.0 / (1.0 + self.entropy)
        self.iterations = iterations

    def _apply_kernel(self, values: torch.Tensor) -> torch.Tensor:
        # Apply the separable Gaussian transport kernel with zero padding
        radius = self.kernel.numel() // 2
        horizontal = F.conv2d(
            values,
            self.kernel.view(1, 1, 1, -1),
            padding=(0, radius),
        )
        return F.conv2d(
            horizontal,
            self.kernel.view(1, 1, -1, 1),
            padding=(radius, 0),
        )

    @staticmethod
    def _generalized_kl(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Sum generalized KL over one flattened raster batch
        tiny = torch.finfo(values.dtype).tiny
        log_ratio_term = torch.where(
            values > 0.0,
            values * (values.clamp_min(tiny).log() - reference.clamp_min(tiny).log()),
            torch.zeros_like(values),
        )
        return (log_ratio_term - values + reference).sum(dim=(1, 2, 3))

    def _regularized_transport(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute an entropy-regularized unbalanced transport cost per raster
        source_mass = source.sum(dim=(1, 2, 3))
        target_mass = target.sum(dim=(1, 2, 3))
        costs = source_mass + target_mass
        active = (source_mass > 0.0) & (target_mass > 0.0)
        if not active.any():
            return costs

        active_source = source[active]
        active_target = target[active]
        with torch.no_grad():
            source_scaling = torch.ones_like(active_source)
            target_scaling = torch.ones_like(active_target)
            for _ in range(self.iterations):
                source_scaling = self._apply_kernel(active_target * target_scaling).clamp_min(1e-12).pow(
                    -self.mass_exponent
                )
                target_scaling = self._apply_kernel(active_source * source_scaling).clamp_min(1e-12).pow(
                    -self.mass_exponent
                )

        scaled_source = active_source * source_scaling
        scaled_target = active_target * target_scaling
        source_marginal = scaled_source * self._apply_kernel(scaled_target)
        target_marginal = scaled_target * self._apply_kernel(scaled_source)
        transported_mass = source_marginal.sum(dim=(1, 2, 3))
        entropy_cost = self.entropy * (
            torch.special.xlogy(source_marginal, source_scaling).sum(dim=(1, 2, 3))
            + torch.special.xlogy(target_marginal, target_scaling).sum(dim=(1, 2, 3))
            - transported_mass
            + source_mass[active] * target_mass[active]
        )
        costs[active] = (
            entropy_cost
            + self._generalized_kl(source_marginal, active_source)
            + self._generalized_kl(target_marginal, active_target)
        )
        return costs

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return one channel-averaged Sinkhorn divergence per sample.

        Args:
            prediction: Nonnegative rasters shaped batch by channel by height by width.
            target: Nonnegative rasters matching ``prediction``.

        Returns:
            Per-sample losses that remain defined for all-zero target channels.
        """
        batch_size, channels, height, width = prediction.shape
        predicted_maps = prediction.float().reshape(batch_size * channels, 1, height, width)
        target_maps = target.float().reshape(batch_size * channels, 1, height, width)
        cross_cost = self._regularized_transport(predicted_maps, target_maps)
        prediction_self_cost = self._regularized_transport(predicted_maps, predicted_maps)
        target_self_cost = self._regularized_transport(target_maps, target_maps)
        prediction_mass = predicted_maps.sum(dim=(1, 2, 3))
        target_mass = target_maps.sum(dim=(1, 2, 3))
        divergence = (
            cross_cost
            - 0.5 * prediction_self_cost
            - 0.5 * target_self_cost
            + 0.5 * self.entropy * (prediction_mass - target_mass).square()
        )
        return divergence.clamp_min(0.0).reshape(batch_size, channels).mean(dim=1)
