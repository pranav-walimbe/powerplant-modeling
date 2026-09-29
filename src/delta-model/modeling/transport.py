"""CPU advection, dispersion, and decay for NO2 transition residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from config import IMG_RANGE, MODEL_IMAGE_CLIP_ABS, MODEL_IMAGE_KEYS, MODEL_MASK_KEYS

SECONDS_PER_HOUR = 3_600.0
METRES_PER_KM = 1_000.0
MIN_NORMALIZATION_SCALE = 1e-12
MIN_GAUSSIAN_SIGMA = 1e-6


@dataclass(frozen=True)
class TransportConfig:
    """Physical parameters for one NO2 transport step."""

    decay_lifetime_hours: float | None = None
    dispersion_sigma_pixels_at_one_hour: float = 0.0
    next_wind_weight: float = 0.5
    wind_speed_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.decay_lifetime_hours is not None and self.decay_lifetime_hours <= 0:
            raise ValueError("decay_lifetime_hours must be positive or None")
        if self.dispersion_sigma_pixels_at_one_hour < 0:
            raise ValueError("dispersion_sigma_pixels_at_one_hour cannot be negative")
        if not 0.0 <= self.next_wind_weight <= 1.0:
            raise ValueError("next_wind_weight must be between zero and one")
        if self.wind_speed_multiplier <= 0:
            raise ValueError("wind_speed_multiplier must be positive")


def _base_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    # Build pixel-center coordinates for align_corners=False sampling
    x = (2.0 * (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5) / width) - 1.0
    y = (2.0 * (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5) / height) - 1.0
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def _sampling_grid(
    wind_u_mps: torch.Tensor,
    wind_v_mps: torch.Tensor,
    elapsed_hours: torch.Tensor,
) -> torch.Tensor:
    # Backtrace each arrival pixel on the north-up AOI grid
    batch_size, _, height, width = wind_u_mps.shape
    base_grid = _base_grid(height, width, wind_u_mps).expand(batch_size, -1, -1, -1)
    seconds = elapsed_hours[:, None, None, None] * SECONDS_PER_HOUR
    cell_width_metres = IMG_RANGE * METRES_PER_KM / width
    cell_height_metres = IMG_RANGE * METRES_PER_KM / height
    column_displacement = wind_u_mps * seconds / cell_width_metres
    northward_displacement = wind_v_mps * seconds / cell_height_metres
    source_x = base_grid[..., 0] - 2.0 * column_displacement[:, 0] / width
    source_y = base_grid[..., 1] + 2.0 * northward_displacement[:, 0] / height
    return torch.stack((source_x, source_y), dim=-1)


def estimate_edge_background(source_no2: torch.Tensor) -> torch.Tensor:
    """Estimate one robust background value per raster from its perimeter.

    Args:
        source_no2: Physical NO2 shaped batch by one by height by width.

    Returns:
        Median perimeter value shaped batch by one by one by one.
    """
    top = source_no2[:, :, 0, :]
    bottom = source_no2[:, :, -1, :]
    left = source_no2[:, :, 1:-1, 0]
    right = source_no2[:, :, 1:-1, -1]
    perimeter = torch.cat((top, bottom, left, right), dim=-1)
    return perimeter.median(dim=-1).values[:, :, None, None]


def _per_sample_gaussian_blur(values: torch.Tensor, sigma_pixels: torch.Tensor) -> torch.Tensor:
    # Apply separable Gaussian kernels with one width per batch item
    maximum_sigma = float(sigma_pixels.max().item())
    if maximum_sigma <= MIN_GAUSSIAN_SIGMA:
        return values
    radius = max(1, math.ceil(3.0 * maximum_sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=values.dtype, device=values.device)
    safe_sigma = sigma_pixels.clamp_min(MIN_GAUSSIAN_SIGMA)
    kernels = torch.exp(-0.5 * (offsets[None, :] / safe_sigma[:, None]).square())
    kernels = kernels / kernels.sum(dim=1, keepdim=True)
    identity = (offsets == 0).to(values.dtype).expand_as(kernels)
    kernels = torch.where((sigma_pixels > MIN_GAUSSIAN_SIGMA)[:, None], kernels, identity)

    batch_size = values.shape[0]
    grouped = values.transpose(0, 1)
    grouped = F.pad(grouped, (radius, radius, 0, 0))
    grouped = F.conv2d(grouped, kernels[:, None, None, :], groups=batch_size)
    grouped = F.pad(grouped, (0, 0, radius, radius))
    grouped = F.conv2d(grouped, kernels[:, None, :, None], groups=batch_size)
    return grouped.transpose(0, 1)


def advect_no2(
    source_no2: torch.Tensor,
    source_quality: torch.Tensor,
    current_wind_mps: torch.Tensor,
    next_wind_mps: torch.Tensor,
    elapsed_hours: torch.Tensor,
    config: TransportConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advect physical NO2 and its quality mask with blended endpoint winds.

    Args:
        source_no2: Physical NO2 shaped batch by one by height by width.
        source_quality: Source quality with the same shape as ``source_no2``.
        current_wind_mps: Current u and v winds shaped batch by two by height by width.
        next_wind_mps: Next u and v winds with the same shape as ``current_wind_mps``.
        elapsed_hours: One elapsed interval per batch item.
        config: Transport parameters controlling wind blending and speed.

    Returns:
        Advected NO2 and advected source quality.
    """
    tensors = (source_no2, source_quality, current_wind_mps, next_wind_mps, elapsed_hours)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("Transport operations require CPU tensors")
    if source_no2.ndim != 4 or source_no2.shape[1] != 1:
        raise ValueError("source_no2 must have shape batch by one by height by width")
    if source_quality.shape != source_no2.shape:
        raise ValueError("source_quality must match source_no2")
    expected_wind_shape = (source_no2.shape[0], 2, *source_no2.shape[-2:])
    if current_wind_mps.shape != expected_wind_shape or next_wind_mps.shape != expected_wind_shape:
        raise ValueError("wind tensors must have shape batch by two by height by width")
    if elapsed_hours.shape != (source_no2.shape[0],):
        raise ValueError("elapsed_hours must contain one value per batch item")

    next_weight = config.next_wind_weight
    wind = ((1.0 - next_weight) * current_wind_mps + next_weight * next_wind_mps)
    wind = wind * config.wind_speed_multiplier
    grid = _sampling_grid(wind[:, :1], wind[:, 1:], elapsed_hours)
    transported = F.grid_sample(
        source_no2,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    transported_quality = F.grid_sample(
        source_quality,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return transported, transported_quality.clamp(0.0, 1.0)


def disperse_no2(
    transported_no2: torch.Tensor,
    transported_quality: torch.Tensor,
    background: torch.Tensor,
    elapsed_hours: torch.Tensor,
    config: TransportConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply elapsed-time-scaled horizontal Gaussian dispersion.

    Args:
        transported_no2: Advected physical NO2 raster batch.
        transported_quality: Advected source quality raster batch.
        background: One background value per raster.
        elapsed_hours: One elapsed interval per batch item.
        config: Transport parameters controlling dispersion width.

    Returns:
        Dispersed NO2 and dispersed quality.
    """
    sigma = config.dispersion_sigma_pixels_at_one_hour * elapsed_hours.clamp_min(0.0).sqrt()
    if config.dispersion_sigma_pixels_at_one_hour <= MIN_GAUSSIAN_SIGMA:
        return transported_no2, transported_quality
    anomaly = transported_no2 - background
    dispersed = _per_sample_gaussian_blur(anomaly, sigma) + background
    dispersed_quality = _per_sample_gaussian_blur(transported_quality, sigma).clamp(0.0, 1.0)
    return dispersed, dispersed_quality


def decay_no2(
    transported_no2: torch.Tensor,
    background: torch.Tensor,
    elapsed_hours: torch.Tensor,
    config: TransportConfig,
) -> torch.Tensor:
    """Apply exponential decay to enhancement above the perimeter background.

    Args:
        transported_no2: Advected and dispersed physical NO2 raster batch.
        background: One background value per raster.
        elapsed_hours: One elapsed interval per batch item.
        config: Transport parameters controlling effective lifetime.

    Returns:
        Background-preserving decayed NO2.
    """
    if config.decay_lifetime_hours is None:
        return transported_no2
    decay = torch.exp(-elapsed_hours[:, None, None, None] / config.decay_lifetime_hours)
    return background + decay * (transported_no2 - background)


def predict_no2_transport(
    source_no2: torch.Tensor,
    source_quality: torch.Tensor,
    current_wind_mps: torch.Tensor,
    next_wind_mps: torch.Tensor,
    elapsed_hours: torch.Tensor,
    config: TransportConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply horizontal advection, dispersion, and effective decay.

    Args:
        source_no2: Physical NO2 shaped batch by one by height by width.
        source_quality: Source quality with the same shape as ``source_no2``.
        current_wind_mps: Current u and v winds shaped batch by two by height by width.
        next_wind_mps: Next u and v winds with the same shape as ``current_wind_mps``.
        elapsed_hours: One elapsed interval per batch item.
        config: Physical transport configuration.

    Returns:
        Predicted physical NO2 and transported source quality.
    """
    background = estimate_edge_background(source_no2)
    transported, quality = advect_no2(
        source_no2,
        source_quality,
        current_wind_mps,
        next_wind_mps,
        elapsed_hours,
        config,
    )
    transported, quality = disperse_no2(transported, quality, background, elapsed_hours, config)
    return decay_no2(transported, background, elapsed_hours, config), quality


class TransportTransform:
    """Build four transport residuals from five CPU raster observations."""

    def __init__(
        self,
        image_center: tuple[float, ...],
        image_scale: tuple[float, ...],
        *,
        residual_center: float = 0.0,
        residual_scale: float = 1.0,
        config: TransportConfig | None = None,
    ) -> None:
        self.no2_channel = MODEL_IMAGE_KEYS.index("no2")
        self.wind_u_channel = MODEL_IMAGE_KEYS.index("wind_u_80m_mps")
        self.wind_v_channel = MODEL_IMAGE_KEYS.index("wind_v_80m_mps")
        self.quality_channel = len(MODEL_IMAGE_KEYS) + MODEL_MASK_KEYS.index("no2_mask")
        self.wind_u_center = float(image_center[self.wind_u_channel])
        self.wind_u_scale = float(image_scale[self.wind_u_channel])
        self.wind_v_center = float(image_center[self.wind_v_channel])
        self.wind_v_scale = float(image_scale[self.wind_v_channel])
        self.residual_center = float(residual_center)
        self.residual_scale = max(float(residual_scale), MIN_NORMALIZATION_SCALE)
        self.config = config or TransportConfig()

    def _physical_wind(self, image: torch.Tensor) -> torch.Tensor:
        # Restore physical endpoint winds from normalized observation channels
        wind_u = image[:, :, self.wind_u_channel] * self.wind_u_scale + self.wind_u_center
        wind_v = image[:, :, self.wind_v_channel] * self.wind_v_scale + self.wind_v_center
        return torch.stack((wind_u, wind_v), dim=2)

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
            transport_wind_mps: Optional interval wind with u and v channels.

        Returns:
            Physical residual and quality tensors with four timesteps.
        """
        inputs = (image, elapsed_hours, no2_normalization)
        if any(value.device.type != "cpu" for value in inputs):
            raise ValueError("TransportTransform requires CPU tensors")
        if transport_wind_mps is not None and transport_wind_mps.device.type != "cpu":
            raise ValueError("TransportTransform requires CPU transport wind tensors")
        batch_size, timesteps, _, height, width = image.shape
        interval_count = timesteps - 1
        no2_center = no2_normalization[:, 0, None, None, None]
        no2_scale = no2_normalization[:, 1, None, None, None]
        physical_no2 = image[:, :, self.no2_channel] * no2_scale + no2_center
        prior = physical_no2[:, :-1].reshape(batch_size * interval_count, 1, height, width)
        current = physical_no2[:, 1:].reshape_as(prior)

        if transport_wind_mps is None:
            endpoint_wind = self._physical_wind(image)
            current_wind = endpoint_wind[:, :-1].reshape(batch_size * interval_count, 2, height, width)
            next_wind = endpoint_wind[:, 1:].reshape_as(current_wind)
        else:
            current_wind = transport_wind_mps.reshape(batch_size * interval_count, 2, height, width)
            next_wind = current_wind

        quality = image[:, :, self.quality_channel : self.quality_channel + 1]
        source_quality = quality[:, :-1].reshape_as(prior)
        current_quality = quality[:, 1:].reshape_as(prior)
        interval_hours = elapsed_hours.reshape(-1)
        transported, transported_quality = predict_no2_transport(
            prior,
            source_quality,
            current_wind,
            next_wind,
            interval_hours,
            self.config,
        )
        residual_quality = (current_quality * transported_quality).clamp(0.0, 1.0)
        residual = (current - transported).reshape(batch_size, interval_count, 1, height, width)
        return residual, residual_quality.reshape_as(residual)

    def __call__(
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
