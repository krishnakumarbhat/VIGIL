"""Test-time-trained convolutional cell model ("neuro" half of the solver).

A very small fully-convolutional network is trained from scratch on the training
pairs of a *single* ARC task, with D4 augmentation.  It is accepted only when it
reproduces every training cell exactly.  The dilated stack gives a receptive
field of about 15 cells, enough for the local-to-regional rules that dominate
same-shape ARC tasks, while keeping one task's fit well under a second on GPU.

This module is deliberately dependency-light: it imports torch lazily so the CPU
search path never pays for it.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src import dsl_primitives as dsl
from src.canonical import D4

Grid = np.ndarray
Pair = Tuple[Grid, Grid]


class CnnCellLearner:
    """Per-task CNN that maps a colour grid to a colour grid of equal shape."""

    def __init__(self, max_seconds: float = 3.0, steps: int = 250,
                 device: Optional[str] = None) -> None:
        """Configure the training budget and target device.

        ``device=None`` picks CUDA when available; the runner passes ``"cuda"``
        explicitly for the sequential post-pass and ``"cpu"`` inside workers.
        """
        self.max_seconds = max_seconds
        self.steps = steps
        self.device = device

    # -- tensor encoding ---------------------------------------------------

    @staticmethod
    def _encode(grids: Sequence[Grid], shape: Tuple[int, int]) -> Any:
        """Encode grids as (N, 13, H, W): 10 colours + row + col + validity."""
        import torch

        h, w = shape
        batch = np.zeros((len(grids), 13, h, w), dtype=np.float32)
        rr = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        cc = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        for index, grid in enumerate(grids):
            if grid.shape != shape:
                raise ValueError("all grids in a level must share one shape")
            batch[index, grid.ravel(), np.repeat(np.arange(h), w), np.tile(np.arange(w), h)] = 1.0
            batch[index, 10, :, :] = rr
            batch[index, 11, :, :] = cc
            batch[index, 12, :, :] = 1.0
        return torch.from_numpy(batch)

    # -- model -------------------------------------------------------------

    @staticmethod
    def _build() -> Any:
        """Build the dilated fully-convolutional network."""
        from torch import nn

        return nn.Sequential(
            nn.Conv2d(13, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(64, 10, 1),
        )

    # -- public API --------------------------------------------------------

    def fit(self, pairs: Sequence[Pair]) -> Optional[Dict[str, Any]]:
        """Train on one task's pairs; return a bundle only on a perfect fit."""
        try:
            import torch
            from torch import nn
        except Exception:
            return None
        if not pairs:
            return None
        shape = pairs[0][0].shape
        if any(src.shape != shape or dst.shape != shape for src, dst in pairs):
            return None
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            views = [fn for _, fn in D4]
            sources: List[Grid] = []
            targets: List[Grid] = []
            for src, dst in pairs:
                for view in views:
                    sources.append(view(src))
                    targets.append(view(dst))
            x = self._encode(sources, shape).to(device)
            y = torch.from_numpy(
                np.stack([t.ravel() for t in targets]).astype(np.int64)
            ).reshape(len(targets), shape[0], shape[1]).to(device)
        except Exception:
            return None
        model = self._build().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        loss_fn = nn.CrossEntropyLoss()
        start = time.monotonic()
        model.train()
        for _ in range(self.steps):
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            if time.monotonic() - start > self.max_seconds:
                break
            if float(loss.detach()) < 1e-4:
                break
        # Exactness gate on the *un-augmented* pairs.
        model.eval()
        with torch.no_grad():
            for src, dst in pairs:
                logits = model(self._encode([src], shape).to(device))
                pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int8)
                if not np.array_equal(pred, dst):
                    return None
        return {"model": model, "shape": shape, "device": device}

    def predict(self, bundle: Dict[str, Any], grid: Grid) -> Optional[Grid]:
        """Predict one grid; ``None`` when the shape differs from training."""
        import torch

        if grid.shape != bundle["shape"]:
            return None
        model = bundle["model"]
        model.eval()
        with torch.no_grad():
            logits = model(self._encode([grid], grid.shape).to(bundle["device"]))
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int8)
        return dsl.as_grid(pred)
