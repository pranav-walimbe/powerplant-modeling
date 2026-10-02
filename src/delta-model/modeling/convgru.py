"""Convolutional recurrent network for emissions-change classification."""

import torch
from modeling.mlp import SEASONAL_EMBEDDING_DIM, build_seasonal_encoder
from torch import nn
from torch.nn import functional as F

from config import MODEL_CLASS_NAMES, MODEL_IMAGE_KEYS, MODEL_MASK_KEYS

DEFAULT_HEAD_DIM = 64
DEFAULT_DROPOUT = 0.20
DEFAULT_MODEL_WIDTH = 64
DEFAULT_CONTEXT_DROPOUT = 0.25


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


class VisionClassifier(nn.Module):
    """Classify change from visual inputs with context at every timestep."""

    def __init__(
        self,
        n_features: int,
        *,
        model_width: int = DEFAULT_MODEL_WIDTH,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
        context_dropout: float = DEFAULT_CONTEXT_DROPOUT,
    ) -> None:
        super().__init__()
        self.context_dropout = context_dropout
        self.observation_encoder = BranchFrameEncoder(5, model_width)
        self.residual_encoder = BranchFrameEncoder(2, model_width)
        self.context_encoder = build_seasonal_encoder(n_features)
        self.observation_film_projection = nn.Linear(SEASONAL_EMBEDDING_DIM, 2 * model_width)
        self.residual_film_projection = nn.Linear(SEASONAL_EMBEDDING_DIM, 2 * model_width)
        nn.init.zeros_(self.observation_film_projection.weight)
        nn.init.zeros_(self.observation_film_projection.bias)
        nn.init.zeros_(self.residual_film_projection.weight)
        nn.init.zeros_(self.residual_film_projection.bias)
        self.branch_fusion = nn.Sequential(
            nn.Conv2d(2 * model_width, model_width, kernel_size=1, bias=False),
            _group_norm(model_width),
            nn.SiLU(inplace=True),
        )
        self.temporal_encoder = ConvGRUCell(model_width, model_width)
        self.temporal_projection = nn.Sequential(
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

    def _encode_frames(
        self,
        encoder: BranchFrameEncoder,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        # Encode every frame with the selected visual branch
        batch_size, timesteps, channels, height, width = sequence.shape
        encoded = encoder(sequence.reshape(batch_size * timesteps, channels, height, width))
        return encoded.reshape(batch_size, timesteps, *encoded.shape[1:])

    def _aggregate_aligned(
        self,
        observation: torch.Tensor,
        residual: torch.Tensor,
        observation_scale: torch.Tensor,
        observation_shift: torch.Tensor,
        residual_scale: torch.Tensor,
        residual_shift: torch.Tensor,
    ) -> torch.Tensor:
        # Return the final recurrent map without synthesizing an initial residual
        hidden = observation[:, 0] * observation_scale[:, 0, :, None, None] + observation_shift[:, 0, :, None, None]
        for transition in range(residual.shape[1]):
            observation_features = (
                observation[:, transition + 1] * observation_scale[:, transition + 1, :, None, None]
                + observation_shift[:, transition + 1, :, None, None]
            )
            residual_features = (
                residual[:, transition] * residual_scale[:, transition, :, None, None]
                + residual_shift[:, transition, :, None, None]
            )
            aligned = torch.cat(
                (observation_features, residual_features),
                dim=1,
            )
            hidden = self.temporal_encoder(self.branch_fusion(aligned), hidden)
        return hidden

    def _project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        # Pool the final recurrent map for classification
        average = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        peak = F.adaptive_max_pool2d(hidden, 1).flatten(1)
        return self.temporal_projection(torch.cat((average, peak), dim=1))

    def _context_film(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep dropout consistent across every timestep in a sample
        embedding = self.context_encoder(context)
        if self.training and self.context_dropout > 0.0:
            keep = (torch.rand(context.shape[0], 1, 1, device=context.device) >= self.context_dropout).to(context.dtype)
            embedding = embedding * keep
        observation_scale_delta, observation_shift = self.observation_film_projection(embedding).chunk(2, dim=-1)
        residual_scale_delta, residual_shift = self.residual_film_projection(embedding[:, 1:]).chunk(2, dim=-1)
        return (
            1.0 + observation_scale_delta,
            observation_shift,
            1.0 + residual_scale_delta,
            residual_shift,
        )

    def forward(
        self,
        image: torch.Tensor,
        seasonal: torch.Tensor,
        transport: torch.Tensor,
    ) -> torch.Tensor:
        observation_features = self._encode_frames(
            self.observation_encoder,
            self._observation_sequence(image),
        )
        residual_features = self._encode_frames(
            self.residual_encoder,
            transport,
        )
        observation_scale, observation_shift, residual_scale, residual_shift = self._context_film(seasonal)
        hidden = self._aggregate_aligned(
            observation_features,
            residual_features,
            observation_scale,
            observation_shift,
            residual_scale,
            residual_shift,
        )
        return self.classifier(self._project_hidden(hidden))

    def num_params(self) -> int:
        """Count trainable model parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class JointVisionClassifier(VisionClassifier):
    """Predict the change class and an auxiliary 12x12 delta raster."""

    def __init__(
        self,
        n_features: int,
        *,
        model_width: int = DEFAULT_MODEL_WIDTH,
        head_dim: int = DEFAULT_HEAD_DIM,
        dropout: float = DEFAULT_DROPOUT,
        context_dropout: float = DEFAULT_CONTEXT_DROPOUT,
    ) -> None:
        super().__init__(
            n_features,
            model_width=model_width,
            head_dim=head_dim,
            dropout=dropout,
            context_dropout=context_dropout,
        )
        decoder_width = model_width // 2
        self.raster_decoder = nn.Sequential(
            ResidualBlock(model_width, model_width),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(model_width, decoder_width, kernel_size=3, padding=1, bias=False),
            _group_norm(decoder_width),
            nn.SiLU(inplace=True),
            ResidualBlock(decoder_width, decoder_width),
            nn.Conv2d(decoder_width, 1, kernel_size=1),
        )

    def forward(
        self,
        image: torch.Tensor,
        seasonal: torch.Tensor,
        transport: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and the normalized auxiliary raster."""
        observation_features = self._encode_frames(
            self.observation_encoder,
            self._observation_sequence(image),
        )
        residual_features = self._encode_frames(
            self.residual_encoder,
            transport,
        )
        observation_scale, observation_shift, residual_scale, residual_shift = self._context_film(seasonal)
        hidden = self._aggregate_aligned(
            observation_features,
            residual_features,
            observation_scale,
            observation_shift,
            residual_scale,
            residual_shift,
        )
        logits = self.classifier(self._project_hidden(hidden))
        return logits, self.raster_decoder(hidden).squeeze(1)
