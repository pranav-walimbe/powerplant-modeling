"""Differentiable GPU transport for NO2 transition residuals."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config import IMG_RANGE, MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_KEYS, MODEL_MASK_KEYS

SECONDS_PER_HOUR = 3_600.0
METRES_PER_KM = 1_000.0
MIN_NORMALIZATION_SCALE = 1e-12


class AdvectionTransport(nn.Module):
    """Build four physical transition residuals from five raster observations."""

    def __init__(
        self,
        image_center: tuple[float, ...],
        image_scale: tuple[float, ...],
        *,
        residual_center: float = 0.0,
        residual_scale: float = 1.0,
        decay_lifetime_hours: float | None = None,
    ) -> None:
        super().__init__()
        self.no2_channel = MODEL_IMAGE_KEYS.index("no2")
        self.wind_u_channel = MODEL_IMAGE_KEYS.index("wind_u_80m_mps")
        self.wind_v_channel = MODEL_IMAGE_KEYS.index("wind_v_80m_mps")
        self.quality_channel = len(MODEL_IMAGE_KEYS) + MODEL_MASK_KEYS.index("no2_mask")
        self.register_buffer("wind_u_center", torch.tensor(float(image_center[self.wind_u_channel])))
        self.register_buffer("wind_u_scale", torch.tensor(float(image_scale[self.wind_u_channel])))
        self.register_buffer("wind_v_center", torch.tensor(float(image_center[self.wind_v_channel])))
        self.register_buffer("wind_v_scale", torch.tensor(float(image_scale[self.wind_v_channel])))
        self.register_buffer("residual_center", torch.tensor(float(residual_center)))
        self.register_buffer("residual_scale", torch.tensor(max(float(residual_scale), MIN_NORMALIZATION_SCALE)))
        lifetime = float("nan") if decay_lifetime_hours is None else float(decay_lifetime_hours)
        self.register_buffer("decay_lifetime_hours", torch.tensor(lifetime))

    @staticmethod
    def _base_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        # Build pixel-center coordinates for align_corners=False sampling
        x = (2.0 * (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5) / width) - 1.0
        y = (2.0 * (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5) / height) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    def _sampling_grid(
        self,
        wind_u_mps: torch.Tensor,
        wind_v_mps: torch.Tensor,
        elapsed_hours: torch.Tensor,
    ) -> torch.Tensor:
        # Backtrace each arrival pixel on the north-up AOI grid
        batch_intervals, _, height, width = wind_u_mps.shape
        base_grid = self._base_grid(height, width, wind_u_mps).expand(batch_intervals, -1, -1, -1)
        seconds = elapsed_hours[:, None, None, None] * SECONDS_PER_HOUR
        cell_width_metres = IMG_RANGE * METRES_PER_KM / width
        cell_height_metres = IMG_RANGE * METRES_PER_KM / height
        column_displacement = wind_u_mps * seconds / cell_width_metres
        northward_displacement = wind_v_mps * seconds / cell_height_metres
        source_x = base_grid[..., 0] - 2.0 * column_displacement[:, 0] / width
        source_y = base_grid[..., 1] + 2.0 * northward_displacement[:, 0] / height
        return torch.stack((source_x, source_y), dim=-1)

    def _interval_wind(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Average physical endpoint winds over each scan interval
        wind_u = image[:, :, self.wind_u_channel : self.wind_u_channel + 1]
        wind_v = image[:, :, self.wind_v_channel : self.wind_v_channel + 1]
        wind_u = wind_u * self.wind_u_scale + self.wind_u_center
        wind_v = wind_v * self.wind_v_scale + self.wind_v_center
        return 0.5 * (wind_u[:, :-1] + wind_u[:, 1:]), 0.5 * (wind_v[:, :-1] + wind_v[:, 1:])

    def physical_residual(
        self,
        image: torch.Tensor,
        elapsed_hours: torch.Tensor,
        no2_normalization: torch.Tensor,
        *,
        transport_wind_mps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate physical residuals and mask-derived quality rasters.

        Args:
            image: Completed raster sequence shaped batch by five by channel by height by width.
            elapsed_hours: Four elapsed scan intervals per sample.
            no2_normalization: Per-sample NO2 center and scale used by the observation branch.
            transport_wind_mps: Optional interval wind with u and v channels for elevated-wind experiments.

        Returns:
            Physical residual and quality tensors with four timesteps.
        """
        batch_size, timesteps, _, height, width = image.shape
        no2_center = no2_normalization[:, 0, None, None, None]
        no2_scale = no2_normalization[:, 1, None, None, None]
        physical_no2 = image[:, :, self.no2_channel] * no2_scale + no2_center
        prior = physical_no2[:, :-1].reshape(batch_size * (timesteps - 1), 1, height, width)
        current = physical_no2[:, 1:].reshape_as(prior)

        if transport_wind_mps is None:
            wind_u, wind_v = self._interval_wind(image)
        else:
            wind_u = transport_wind_mps[:, :, :1]
            wind_v = transport_wind_mps[:, :, 1:2]
        wind_u = wind_u.reshape_as(prior)
        wind_v = wind_v.reshape_as(prior)
        interval_hours = elapsed_hours.reshape(-1)
        grid = self._sampling_grid(wind_u, wind_v, interval_hours)
        transported = F.grid_sample(
            prior,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        if torch.isfinite(self.decay_lifetime_hours):
            decay = torch.exp(-interval_hours[:, None, None, None] / self.decay_lifetime_hours)
            transported = decay * transported

        quality = image[:, :, self.quality_channel : self.quality_channel + 1]
        source_quality = quality[:, :-1].reshape_as(prior)
        transported_quality = F.grid_sample(
            source_quality,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        current_quality = quality[:, 1:].reshape_as(prior)
        residual_quality = (current_quality * transported_quality).clamp(0.0, 1.0)
        residual = (current - transported).reshape(batch_size, timesteps - 1, 1, height, width)
        return residual, residual_quality.reshape_as(residual)

    def forward(
        self,
        image: torch.Tensor,
        elapsed_hours: torch.Tensor,
        no2_normalization: torch.Tensor,
        *,
        transport_wind_mps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized residuals and their mask-derived quality."""
        residual, quality = self.physical_residual(
            image,
            elapsed_hours,
            no2_normalization,
            transport_wind_mps=transport_wind_mps,
        )
        normalized = ((residual - self.residual_center) / self.residual_scale).clamp(
            -MODEL_IMAGE_CLIP_ABS,
            MODEL_IMAGE_CLIP_ABS,
        )
        return normalized, quality
