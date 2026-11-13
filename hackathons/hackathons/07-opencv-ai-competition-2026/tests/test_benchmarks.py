"""Metric-helper tests for the offline benchmark harness."""

from __future__ import annotations

import math

from benchmarks.run_eval import iou, match_faces, percentile, prf, roc_auc


def test_percentile_edges() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile([], 50) == 0.0
    assert percentile(values, 50) == 2.0
    assert percentile([5.0], 95) == 5.0
    assert percentile(values, 0) == 1.0


def test_roc_auc_perfect_and_tied() -> None:
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    assert math.isnan(roc_auc([0.2], [1]))


def test_iou_and_prf() -> None:
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0
    assert prf(2, 0, 1) == {"precision": 1.0, "recall": 0.6667, "f1": 0.8}
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_match_faces_counts_identity_only_for_known_gt() -> None:
    gt = [
        {"box": [10, 10, 50, 50], "known": True, "id": "alice"},
        {"box": [100, 100, 40, 40], "known": False, "id": None},
    ]
    pred = [
        {"box": [11, 11, 50, 50], "identity": "alice"},
        {"box": [101, 101, 40, 40], "identity": "bob"},
        {"box": [200, 10, 30, 30], "identity": None},
    ]
    result = match_faces(gt, pred, threshold=0.35)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["id_correct"] == 1
    assert result["id_total"] == 1
    assert result["mean_iou"] > 0.9
