"""Compact seasonal model for emissions-change classification."""

import torch
from torch import nn

from config import MODEL_CLASS_NAMES

SEASONAL_HIDDEN_DIM = 16
SEASONAL_EMBEDDING_DIM = 8


def build_seasonal_encoder(
    n_features: int,
    hidden_dim: int = SEASONAL_HIDDEN_DIM,
    embedding_dim: int = SEASONAL_EMBEDDING_DIM,
) -> nn.Sequential:
    """Build the shared seasonal and solar context encoder."""
    return nn.Sequential(
        nn.Linear(n_features, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, embedding_dim),
        nn.LayerNorm(embedding_dim),
        nn.SiLU(inplace=True),
    )


class SeasonalMLP(nn.Module):
    """Classify emissions changes from daily and annual phase."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = SEASONAL_HIDDEN_DIM,
        embedding_dim: int = SEASONAL_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = build_seasonal_encoder(n_features, hidden_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, len(MODEL_CLASS_NAMES))

    def encode(self, seasonal: torch.Tensor) -> torch.Tensor:
        """Produce the embedding used by the classification head."""
        return self.encoder(seasonal)

    def forward(
        self,
        image: torch.Tensor,
        seasonal: torch.Tensor,
        transport: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one logit for each emissions-change class."""
        del image, transport
        return self.classifier(self.encode(seasonal))

    def num_params(self) -> int:
        """Count trainable model parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
