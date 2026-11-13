#!/usr/bin/env python3
"""Re-runnable submission builder from saved artifacts."""
import os, numpy as np, pandas as pd
from scipy.special import expit, logit
ROOT = "/home/pope/Desktop/gpu /hackathons/06-playground-s6e10-airline-satisfaction"
OUT = os.path.join(ROOT, "outputs"); DATA = os.path.join(ROOT, "data")
te = pd.read_csv(os.path.join(DATA, "test.csv"))
# Load weights
import json
w = json.load(open(os.path.join(OUT, "cv_scores.json")))["weights"]
keys = list(w.keys())
preds = {}
for k in keys:
    pth = os.path.join(OUT, f"test_{k}.npy")
    if os.path.exists(pth):
        preds[k] = np.load(pth)
    else:
        preds[k] = np.ones(len(te)) * 0.5  # placeholder if missing
blend = np.zeros(len(te))
for k in keys:
    blend += w[k] * logit(np.clip(preds[k], 1e-6, 1-1e-6))
final = expit(blend / sum(w.values()))
pd.DataFrame({"id": te["id"], "satisfaction": final}).to_csv(os.path.join(OUT, "submission.csv"), index=False)
print("submission.csv rebuilt, rows=", len(te), "mean=", final.mean())
