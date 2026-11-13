# Track 08: OpenZee OpenAir 2026 — Modular Multimodal Starter Template

## Architecture Overview
Following repository coding standards (1 class per file, full type annotations, docstrings), the starter architecture is structured as follows:
- `src/dataset.py`: `MultimodalDataset` — extensible tabular, sequence, and visual dataset container.
- `src/model.py`: `MultimodalClassifier` — modular feed-forward feature encoder with classification head.
- `src/trainer.py`: `Trainer` — self-contained training and validation harness.

## Activation Instructions
Once the official track specification or real dataset is acquired:
1. Parse raw data into tabular arrays, text tokens, or image tensors.
2. Initialize `MultimodalDataset` and wrap with PyTorch `DataLoader`.
3. Instantiate `MultimodalClassifier` with the appropriate input dimension and class count.
4. Execute `Trainer.train_epoch()` and `Trainer.evaluate()` within your cross-validation loop.

## Smoke test (measured)
A 64-row synthetic run (12 features, 4 classes, batch 16, AdamW lr=1e-2, CPU) trains and
evaluates end to end: loss 1.456 → 1.276, accuracy 0.500 on a single epoch.
All three modules are type-hinted, one class per file, and import with no side effects.
