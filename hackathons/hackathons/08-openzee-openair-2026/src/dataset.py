"""Multimodal tabular, text, and visual dataset module for Track 08."""

from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """Encapsulates multimodal features including numeric tabular arrays, token sequences, and image tensors."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        image_tensors: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the MultimodalDataset.

        Args:
            features: Tensor of continuous tabular features of shape (N, D).
            labels: Optional ground-truth label tensor of shape (N,).
            text_tokens: Optional tokenized text sequences of shape (N, L).
            image_tensors: Optional image feature or pixel tensors of shape (N, C, H, W).
        """
        self.features: torch.Tensor = features
        self.labels: Optional[torch.Tensor] = labels
        self.text_tokens: Optional[torch.Tensor] = text_tokens
        self.image_tensors: Optional[torch.Tensor] = image_tensors

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch item at specified index."""
        item: Dict[str, torch.Tensor] = {
            "features": self.features[idx]
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.text_tokens is not None:
            item["text_tokens"] = self.text_tokens[idx]
        if self.image_tensors is not None:
            item["image_tensors"] = self.image_tensors[idx]
        return item
