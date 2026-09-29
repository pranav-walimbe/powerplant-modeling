"""Convolutional recurrent network for emissions-change classification."""

import torch
from modeling.mlp import SEASONAL_EMBEDDING_DIM, SeasonalMLP
from modeling.transport import AdvectionTransport
from torch import nn
from torch.nn import functional as F

from config import MODEL_CLASS_NAMES, MODEL_IMAGE_CHANNELS, MODEL_IMAGE_KEYS, MODEL_INPUT_CHANNELS, MODEL_MASK_KEYS

DEFAULT_HEAD_DIM = 64
DEFAULT_DROPOUT = 0.20
DEFAULT_MODEL_WIDTH = 64
ENCODER_OUTPUT_CHANNELS = 64
DEFAULT_CONTEXT_DROPOUT = 0.25
MAX_CONTEXT_SCALE_DELTA = 0.25


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """Residual spatial block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
        )
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(inputs) + self.residual(inputs))


class NO2Stem(nn.Module):
    """Encode a completed NO2 raster with ordinary convolutions."""

    def __init__(self, out_channels: int = 16) -> None:
        super().__init__()
        self.first = nn.Conv2d(1, out_channels, kernel_size=5, padding=2)
        self.first_norm = _group_norm(out_channels)
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.second_norm = _group_norm(out_channels)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.first_norm(self.first(values)))
        return self.activation(self.second_norm(self.second(features)))


class RasterFrameEncoder(nn.Module):
    """Encode one completed NO2 and weather frame with ordinary convolutions."""

    def __init__(self, model_width: int = DEFAULT_MODEL_WIDTH) -> None:
        super().__init__()
        stem_channels = model_width // 4
        fusion_channels = model_width // 2
        middle_channels = 3 * model_width // 4
        self.no2_stem = NO2Stem(out_channels=stem_channels)
        self.weather_stem = nn.Sequential(
            nn.Conv2d(MODEL_IMAGE_CHANNELS - 1, stem_channels, kernel_size=5, padding=2, bias=False),
            _group_norm(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.stem_fusion = nn.Sequential(
            nn.Conv2d(2 * stem_channels, fusion_channels, kernel_size=1, bias=False),
            _group_norm(fusion_channels),
            nn.SiLU(inplace=True),
        )
        self.spatial_encoder = nn.Sequential(
            ResidualBlock(fusion_channels, fusion_channels),
            ResidualBlock(fusion_channels, middle_channels, stride=2),
            ResidualBlock(middle_channels, middle_channels),
            ResidualBlock(middle_channels, model_width, stride=2),
            ResidualBlock(model_width, model_width),
        )

    def forward(self, no2: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        no2_features = self.no2_stem(no2)
        weather_features = self.weather_stem(weather)
        fused = self.stem_fusion(torch.cat((no2_features, weather_features), dim=1))
        return self.spatial_encoder(fused)


class ConvGRUCell(nn.Module):
    """Spatial gated recurrent unit that preserves a feature map over time."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        combined_channels = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(combined_channels, 2 * hidden_channels, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(combined_channels, hidden_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((inputs, hidden), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((inputs, reset * hidden), dim=1)).tanh()
        return (1.0 - update) * hidden + update * candidate


class BranchFrameEncoder(nn.Module):
    """Encode one aligned multichannel raster without modality-specific stems."""

    def __init__(self, in_channels: int, model_width: int) -> None:
        super().__init__()
        stem_channels = model_width // 2
        middle_channels = 3 * model_width // 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=5, padding=2, bias=False),
            _group_norm(stem_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(stem_channels, stem_channels),
            ResidualBlock(stem_channels, middle_channels, stride=2),
            ResidualBlock(middle_channels, middle_channels),
            ResidualBlock(middle_channels, model_width, stride=2),
            ResidualBlock(model_width, model_width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)


class TemporalRasterBranch(nn.Module):
    """Encode raster frames and aggregate their spatial features over time."""

    def __init__(self, in_channels: int, model_width: int, dropout: float) -> None:
        super().__init__()
        self.model_width = model_width
        self.frame_encoder = BranchFrameEncoder(in_channels, model_width)
        self.temporal_encoder = ConvGRUCell(model_width, model_width)
        self.projection = nn.Sequential(
            nn.Linear(2 * model_width, model_width),
            nn.LayerNorm(model_width),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor, scale_delta: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = sequence.shape
        encoded = self.frame_encoder(sequence.reshape(batch_size * timesteps, channels, height, width))
        encoded = encoded.reshape(batch_size, timesteps, *encoded.shape[1:])
        if scale_delta is not None:
            encoded = encoded + encoded * scale_delta[:, None, :, None, None]

        hidden = encoded.new_zeros(
            batch_size,
            self.model_width,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        for timestep in range(timesteps):
            hidden = self.temporal_encoder(encoded[:, timestep], hidden)
        average = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        peak = F.adaptive_max_pool2d(hidden, 1).flatten(1)
        return self.projection(torch.cat((average, peak), dim=1))


class ContextScaleConditioner(nn.Module):
    """Produce bounded branch-specific scale changes from sample context."""

    def __init__(
        self,
        n_features: int,
        model_width: int,
        *,
        context_dropout: float = DEFAULT_CONTEXT_DROPOUT,
        max_scale_delta: float = MAX_CONTEXT_SCALE_DELTA,
    ) -> None:
        super().__init__()
        context_width = max(16, 4 * n_features)
        self.context_dropout = context_dropout
        self.max_scale_delta = max_scale_delta
        self.encoder = nn.Sequential(
            nn.Linear(n_features, context_width),
            nn.SiLU(inplace=True),
            nn.Linear(context_width, context_width),
            nn.SiLU(inplace=True),
        )
        self.observation_projection = nn.Linear(context_width, model_width)
        self.residual_projection = nn.Linear(context_width, model_width)
        nn.init.zeros_(self.observation_projection.weight)
        nn.init.zeros_(self.observation_projection.bias)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(context)
        keep = torch.ones(context.shape[0], 1, device=context.device, dtype=context.dtype)
        if self.training and self.context_dropout > 0.0:
            keep = (torch.rand_like(keep) >= self.context_dropout).to(context.dtype)
        observation_delta = self.max_scale_delta * torch.tanh(self.observation_projection(embedding)) * keep
        residual_delta = self.max_scale_delta * torch.tanh(self.residual_projection(embedding)) * keep
        return observation_delta, residual_delta


class RasterConvGRUClassifier(nn.Module):
    """Classify emissions changes from completed raster sequences."""

    def __init__(
        self,
        *,
        model_width: int = DEFAULT_MODEL_WIDTH,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.model_width = model_width
        self.frame_encoder = RasterFrameEncoder(model_width)
        self.temporal_encoder = ConvGRUCell(model_width, model_width)
        self.vision_projection = nn.Sequential(
            nn.Linear(2 * model_width, model_width),
            nn.LayerNorm(model_width),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(model_width, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, len(MODEL_CLASS_NAMES)),
        )

    def _encode_sequence(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = image.shape
        if channels != MODEL_INPUT_CHANNELS:
            raise ValueError(f"Expected {MODEL_INPUT_CHANNELS} completed raster channels, received {channels}")
        frames = image[:, :, :MODEL_IMAGE_CHANNELS].reshape(
            batch_size * timesteps,
            MODEL_IMAGE_CHANNELS,
            height,
            width,
        )
        no2 = frames[:, :1]
        weather = frames[:, 1:MODEL_IMAGE_CHANNELS]
        encoded = self.frame_encoder(no2, weather)
        encoded = encoded.reshape(batch_size, timesteps, *encoded.shape[1:])

        hidden = encoded.new_zeros(
            batch_size,
            self.model_width,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        for timestep in range(timesteps):
            hidden = self.temporal_encoder(encoded[:, timestep], hidden)
        average = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        peak = F.adaptive_max_pool2d(hidden, 1).flatten(1)
        return self.vision_projection(torch.cat((average, peak), dim=1))

    def forward(
        self,
        image: torch.Tensor,
        seasonal: torch.Tensor,
        elapsed_hours: torch.Tensor,
        no2_normalization: torch.Tensor,
    ) -> torch.Tensor:
        del seasonal, elapsed_hours, no2_normalization
        return self.classifier(self._encode_sequence(image))

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class LearnedEmbeddingFusionClassifier(RasterConvGRUClassifier):
    """Jointly classify learned raster and seasonal embeddings."""

    def __init__(
        self,
        n_features: int,
        *,
        model_width: int = DEFAULT_MODEL_WIDTH,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__(model_width=model_width, head_dim=head_dim, dropout=dropout)
        self.seasonal_encoder = SeasonalMLP(n_features).encoder
        self.classifier = nn.Sequential(
            nn.Linear(model_width + SEASONAL_EMBEDDING_DIM, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, len(MODEL_CLASS_NAMES)),
        )

    def forward(
        self,
        image: torch.Tensor,
        seasonal: torch.Tensor,
        elapsed_hours: torch.Tensor,
        no2_normalization: torch.Tensor,
    ) -> torch.Tensor:
        del elapsed_hours, no2_normalization
        vision_embedding = self._encode_sequence(image)
        seasonal_embedding = self.seasonal_encoder(seasonal)
        return self.classifier(torch.cat((vision_embedding, seasonal_embedding), dim=1))


class TransportContextClassifier(nn.Module):
    """Classify change from observation and transport-residual branches."""

    def __init__(
        self,
        n_features: int,
        image_center: tuple[float, ...],
        image_scale: tuple[float, ...],
        residual_center: float,
        residual_scale: float,
        *,
        condition_on_context: bool,
        model_width: int = DEFAULT_MODEL_WIDTH,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
        decay_lifetime_hours: float | None = None,
    ) -> None:
        super().__init__()
        self.transport = AdvectionTransport(
            image_center,
            image_scale,
            residual_center=residual_center,
            residual_scale=residual_scale,
            decay_lifetime_hours=decay_lifetime_hours,
        )
        self.observation_branch = TemporalRasterBranch(5, model_width, dropout)
        self.residual_branch = TemporalRasterBranch(2, model_width, dropout)
        self.context_conditioner = (
            ContextScaleConditioner(n_features, model_width) if condition_on_context else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * model_width, head_dim),
            nn.LayerNorm(head_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, len(MODEL_CLASS_NAMES)),
        )

        self.no2_channel = MODEL_IMAGE_KEYS.index("no2")
        self.temperature_channel = MODEL_IMAGE_KEYS.index("temperature_2m_k")
        self.wind_u_channel = MODEL_IMAGE_KEYS.index("wind_u_80m_mps")
        self.wind_v_channel = MODEL_IMAGE_KEYS.index("wind_v_80m_mps")
        self.quality_channel = len(MODEL_IMAGE_KEYS) + MODEL_MASK_KEYS.index("no2_mask")

    def _observation_sequence(self, image: torch.Tensor) -> torch.Tensor:
        # Order channels as NO2, quality, wind-u, wind-v, and temperature
        indices = (
            self.no2_channel,
            self.quality_channel,
            self.wind_u_channel,
            self.wind_v_channel,
            self.temperature_channel,
        )
        return image[:, :, indices]

    def forward(
        self,
        image: torch.Tensor,
        seasonal: torch.Tensor,
        elapsed_hours: torch.Tensor,
        no2_normalization: torch.Tensor,
        transport_wind_mps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual, residual_quality = self.transport(
            image,
            elapsed_hours,
            no2_normalization,
            transport_wind_mps=transport_wind_mps,
        )
        observation_delta = None
        residual_delta = None
        if self.context_conditioner is not None:
            observation_delta, residual_delta = self.context_conditioner(seasonal)
        observation_embedding = self.observation_branch(
            self._observation_sequence(image),
            observation_delta,
        )
        residual_embedding = self.residual_branch(
            torch.cat((residual, residual_quality), dim=2),
            residual_delta,
        )
        return self.classifier(torch.cat((observation_embedding, residual_embedding), dim=1))

    def num_params(self) -> int:
        """Count trainable model parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
