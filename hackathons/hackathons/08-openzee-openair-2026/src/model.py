"""Multimodal backbone model architecture."""

from typing import Dict, Optional
import torch
import torch.nn as nn


class MultimodalClassifier(nn.Module):
    """Standardized multimodal projection network supporting tabular, text, and visual representations."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize the MultimodalClassifier.

        Args:
            feature_dim: Dimension of input tabular features.
            num_classes: Target number of classification classes.
            hidden_dim: Latent representation dimensionality.
            dropout_rate: Dropout regularization probability.
        """
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass computing unnormalized class logits.

        Args:
            batch: Dictionary containing 'features' and optional multimodal modalities.

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        x = batch["features"]
        h = self.feature_encoder(x)
        logits = self.classifier(h)
        return logits
