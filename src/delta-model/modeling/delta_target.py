"""Auxiliary delta-raster targets for joint model training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from config import MODEL_IMAGE_KEYS, SEQUENCE_TIMESTEPS

DEFAULT_TARGET_SIZE = 12
DEFAULT_SMOOTHING_SIGMA_CELLS = 1.5
DEFAULT_UPWIND_FRACTION = 1.0 / 3.0
DEFAULT_MIN_WIND_SPEED_MPS = 0.5
MIN_SCALE = 1e-12


@dataclass(frozen=True)
class DeltaTargetConfig:
    """Physical and spatial settings for the auxiliary raster target."""

    target_size: int = DEFAULT_TARGET_SIZE
    smoothing_sigma_cells: float = DEFAULT_SMOOTHING_SIGMA_CELLS
    upwind_fraction: float = DEFAULT_UPWIND_FRACTION
    min_wind_speed_mps: float = DEFAULT_MIN_WIND_SPEED_MPS

    def to_metadata(self) -> dict[str, float | int | str]:
        """Describe the target construction for saved run metadata.

        Returns:
            JSON-serializable target settings and fixed methods.
        """
        return {
            **asdict(self),
            "background_method": "per_timestep_wind_relative_upwind_third_median",
            "calm_wind_fallback": "full_raster_median",
            "temporal_method": "late_minus_early_trapezoidal_mean",
            "no2_source": "masked_model_completed_physical_units",
        }


def _grid_coordinates(height: int, width: int, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Return east and north pixel offsets from the raster center
    rows = torch.arange(height, dtype=reference.dtype, device=reference.device) - (height - 1) / 2
    columns = torch.arange(width, dtype=reference.dtype, device=reference.device) - (width - 1) / 2
    east, south = torch.meshgrid(columns, rows, indexing="xy")
    return east, -south


def _upwind_background(
    no2: torch.Tensor,
    wind_u: torch.Tensor,
    wind_v: torch.Tensor,
    config: DeltaTargetConfig,
) -> torch.Tensor:
    # Estimate one background per sample and timestep
    batch_size, timesteps, height, width = no2.shape
    flattened_u = wind_u.reshape(batch_size, timesteps, -1)
    flattened_v = wind_v.reshape(batch_size, timesteps, -1)
    reference_u = torch.nanmedian(flattened_u, dim=-1).values
    reference_v = torch.nanmedian(flattened_v, dim=-1).values
    finite_wind = torch.isfinite(reference_u) & torch.isfinite(reference_v)
    safe_u = torch.where(finite_wind, reference_u, torch.zeros_like(reference_u))
    safe_v = torch.where(finite_wind, reference_v, torch.zeros_like(reference_v))
    speed = torch.hypot(safe_u, safe_v)

    east, north = _grid_coordinates(height, width, no2)
    safe_speed = speed.clamp_min(config.min_wind_speed_mps)
    along_wind = (
        east[None, None] * safe_u[:, :, None, None]
        + north[None, None] * safe_v[:, :, None, None]
    ) / safe_speed[:, :, None, None]
    cutoff = torch.quantile(along_wind.flatten(2), config.upwind_fraction, dim=-1)
    upwind = along_wind <= cutoff[:, :, None, None]
    upwind_values = no2.masked_fill(~upwind, torch.nan)
    upwind_background = torch.nanmedian(upwind_values.flatten(2), dim=-1).values
    full_background = torch.nanmedian(no2.flatten(2), dim=-1).values
    use_upwind = finite_wind & (speed >= config.min_wind_speed_mps) & torch.isfinite(upwind_background)
    return torch.where(use_upwind, upwind_background, full_background)


def _window_means(values: torch.Tensor, elapsed_hours: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Calculate early and late trapezoidal means for five observations
    if values.shape[1] != SEQUENCE_TIMESTEPS or elapsed_hours.shape[1] != SEQUENCE_TIMESTEPS - 1:
        raise ValueError("Delta targets require five observations and four elapsed intervals")
    if not torch.isfinite(elapsed_hours).all() or torch.any(elapsed_hours <= 0):
        raise ValueError("Elapsed intervals must be finite and positive")
    interval_means = (values[:, :-1] + values[:, 1:]) / 2.0
    spatial_dimensions = (1,) * (values.ndim - 2)
    weights = elapsed_hours.reshape(elapsed_hours.shape[0], elapsed_hours.shape[1], *spatial_dimensions)
    early = (interval_means[:, :2] * weights[:, :2]).sum(dim=1) / weights[:, :2].sum(dim=1)
    late = (interval_means[:, 2:] * weights[:, 2:]).sum(dim=1) / weights[:, 2:].sum(dim=1)
    return early, late


def _gaussian_smooth(values: torch.Tensor, sigma: float) -> torch.Tensor:
    # Apply normalized Gaussian smoothing with nearest-edge padding
    if sigma <= 0:
        raise ValueError("Smoothing sigma must be positive")
    radius = max(1, math.ceil(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, dtype=values.dtype, device=values.device)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)[None, None]
    padded = F.pad(values[:, None], (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel_2d).squeeze(1)


class DeltaRasterTargetTransform:
    """Build a smoothed delta target from completed physical NO2 rasters."""

    def __init__(self, config: DeltaTargetConfig | None = None) -> None:
        self.config = config or DeltaTargetConfig()
        self.no2_channel = MODEL_IMAGE_KEYS.index("no2")
        self.wind_u_channel = MODEL_IMAGE_KEYS.index("wind_u_80m_mps")
        self.wind_v_channel = MODEL_IMAGE_KEYS.index("wind_v_80m_mps")

    def __call__(
        self,
        physical_image: torch.Tensor,
        elapsed_hours: torch.Tensor,
    ) -> torch.Tensor:
        """Construct one physical delta target per input sequence.

        Args:
            physical_image: Physical numeric channels shaped batch by time by channel by height by width.
            elapsed_hours: Four elapsed intervals per sample.

        Returns:
            Physical delta targets at the configured output size.
        """
        if physical_image.ndim != 5:
            raise ValueError("Physical images must be a batched raster sequence")
        height, width = physical_image.shape[-2:]
        if height != width or height % self.config.target_size:
            raise ValueError(f"Raster shape {(height, width)} is incompatible with target size {self.config.target_size}")

        no2 = physical_image[:, :, self.no2_channel]
        background = _upwind_background(
            no2,
            physical_image[:, :, self.wind_u_channel],
            physical_image[:, :, self.wind_v_channel],
            self.config,
        )
        anomalies = no2 - background[:, :, None, None]
        early, late = _window_means(anomalies, elapsed_hours)
        target = late - early

        pool_factor = height // self.config.target_size
        target = F.avg_pool2d(target[:, None], pool_factor).squeeze(1)
        target = _gaussian_smooth(target, self.config.smoothing_sigma_cells)
        return target
