# Executable ARC audit — 6 November 2026

Extract the archive and preserve its directory structure. From the extracted root run:

```
python3 03-arc-prize-2026-paper-track/audit_final.py
```

Requirements: Python 3.10+, NumPy, Matplotlib. No GPU, weights, credentials, or network access are needed for the audit. The original LLM reproduction is NOT included or claimed as our invention. Its source is https://www.kaggle.com/code/koushikrudra/failed-in-aimo (version 1, Apache-2.0).

The script performs 34,992 exhaustive invariance assertions, demonstrates the existing key's rotation and shape defects, recounts historical metrics, verifies two scheduler boundary conditions, deduplicates measured latency lines, and simulates 5,000 seeded resampling runs at three explicitly assumed latency scales.

`AUDIT_FINAL.json` contains measured durations, source SHA-256 hashes, recounts, and conditional simulation results. Re-running overwrites this output and the figure. Simulation percentiles are NOT hidden-set confidence bounds, and no hidden accuracy benefit is claimed.

Historical `solved` fields count candidate discovery. See `search.py` and `solver_v2.py`; graded counters measure accuracy. The training and evaluation JSONs come from separate solver runs/versions.

All bundled files concern this competition. Browser/session material, account credentials, personal files, and model weights are excluded.
