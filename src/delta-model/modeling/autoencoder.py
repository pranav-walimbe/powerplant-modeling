"""Checkpoint-compatible masked autoencoder for completing NO2 rasters."""

import torch
from modeling.convgru import ResidualBlock
from torch import nn
from torch.nn import functional as F

from config import MODEL_IMAGE_CHANNELS

ENCODER_OUTPUT_CHANNELS = 64


def _group_norm(channels: int) -> nn.GroupNorm:
    # Match the normalization layout stored in masked-model checkpoints
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _PartialConv2d(nn.Module):
    # Recreate one partial convolution from the masked-model checkpoint
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.kernel_area = kernel_size * kernel_size
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("mask_kernel", torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.convolution(inputs * mask)
        with torch.no_grad():
            support = F.conv2d(mask, self.mask_kernel, padding=self.padding)
            next_mask = (support > 0).to(inputs.dtype)
            scale = self.kernel_area / support.clamp_min(1.0)
        features = (features * scale + self.bias[None, :, None, None]) * next_mask
        return features, next_mask


class _MaskedNO2Stem(nn.Module):
    # Recreate the masked-model NO2 stem for the filling pass
    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.first = _PartialConv2d(1, out_channels, kernel_size=5)
        self.first_norm = _group_norm(out_channels)
        self.second = _PartialConv2d(out_channels, out_channels, kernel_size=3)
        self.second_norm = _group_norm(out_channels)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features, mask = self.first(values, mask)
        features = self.activation(self.first_norm(features)) * mask
        features, mask = self.second(features, mask)
        return self.activation(self.second_norm(features)) * mask


class _MaskedRasterFrameEncoder(nn.Module):
    # Recreate the masked-model encoder layout for checkpoint loading
    def __init__(self) -> None:
        super().__init__()
        self.no2_stem = _MaskedNO2Stem(out_channels=16)
        self.weather_stem = nn.Sequential(
            nn.Conv2d(MODEL_IMAGE_CHANNELS - 1, 16, kernel_size=5, padding=2, bias=False),
            _group_norm(16),
            nn.SiLU(inplace=True),
        )
        self.stem_fusion = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            _group_norm(32),
            nn.SiLU(inplace=True),
        )
        self.spatial_encoder = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 48, stride=2),
            ResidualBlock(48, 48),
            ResidualBlock(48, ENCODER_OUTPUT_CHANNELS, stride=2),
            ResidualBlock(ENCODER_OUTPUT_CHANNELS, ENCODER_OUTPUT_CHANNELS),
        )

    def forward(
        self,
        no2: torch.Tensor,
        weather: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        no2_features = self.no2_stem(no2, visible_mask)
        weather_features = self.weather_stem(weather)
        fused = self.stem_fusion(torch.cat((no2_features, weather_features), dim=1))
        return self.spatial_encoder(fused)


class MaskedNO2Autoencoder(nn.Module):
    """Reconstruct the pretrained masked model used to complete NO2 rasters."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _MaskedRasterFrameEncoder()
        self.bottleneck = ResidualBlock(ENCODER_OUTPUT_CHANNELS, ENCODER_OUTPUT_CHANNELS)
        self.decoder_12 = ResidualBlock(ENCODER_OUTPUT_CHANNELS, 48)
        self.decoder_24 = ResidualBlock(48, 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict the complete NO2 raster."""
        no2 = image[:, :1]
        weather = image[:, 1:MODEL_IMAGE_CHANNELS]
        visible_mask = image[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        latent = self.encoder(no2, weather, visible_mask)
        decoded = self.bottleneck(latent)
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        decoded = self.decoder_12(decoded)
        decoded = F.interpolate(decoded, scale_factor=2, mode="bilinear", align_corners=False)
        return self.output(self.decoder_24(decoded))

    def fill_missing(self, image: torch.Tensor) -> torch.Tensor:
        """Preserve observed NO2 and fill missing pixels with predictions."""
        visible_mask = image[:, MODEL_IMAGE_CHANNELS : MODEL_IMAGE_CHANNELS + 1]
        return image[:, :1] * visible_mask + self(image) * (1.0 - visible_mask)
