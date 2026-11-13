"""Trainer engine for model training and validation."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Execution harness managing epochs, optimization, and validation metrics."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: PyTorch neural network module.
            optimizer: Torch optimizer instance.
            criterion: Loss function module.
            device: Computing device (CPU or CUDA).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run a single training epoch across the dataloader.

        Args:
            dataloader: DataLoader supplying training batches.

        Returns:
            Mean training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch["labels"])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(batch["labels"])
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Run evaluation pass across the validation dataloader.

        Args:
            dataloader: DataLoader supplying validation batches.

        Returns:
            Tuple of (mean validation loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(batch)
                loss = self.criterion(logits, batch["labels"])
                total_loss += loss.item() * len(batch["labels"])
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])
        return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)
