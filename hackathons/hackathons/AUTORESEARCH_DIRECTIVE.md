# Per-iteration directive (autoresearch driver)

You are one iteration of the multi-track Kaggle hackathon marathon. The workspace is
`/home/pope/Desktop/gpu /hackathons/` (NOTE the literal trailing space in "gpu " - always quote paths).

RULES (non-negotiable):
- NEVER submit to Kaggle or Devpost. Predictions live in `<track>/outputs/`.
- Never modify /media/pope/virtbox/a_resume/.
- Kaggle GPU <= 44h/week (check `kaggle kernels list -m` and the quota endpoint; keep 1h reserve).
- Colab Pro sessions must be stopped; `colab exec -f` takes a LOCAL path; wait for DNS after `colab new`.
- Use system python3; never create a .venv. GPU memory limit locally: 3.2GB.

THIS ITERATION:
1. Read MASTER_STATUS.md, autoresearch_research.md, autoresearch_research.jsonl,
   autoresearch_research_strategy_graph.jsonl, strategies.md, ideas.md, experiments/worklog.md.
2. Choose the single highest-potential frontier node (prefer a node whose metric can be moved
   with the currently idle GPU lane; if the Kaggle GPU slots are busy and Colab is busy, do CPU work).
3. Implement it in the track's src/ folder, evaluate with the track's STRICT validation, and
   record the measurement.
4. Log to autoresearch_research.jsonl (commit, metric, value, status, description),
   append the node/edge to autoresearch_research_strategy_graph.jsonl, update ideas.md and
   experiments/worklog.md, and append to `<track>/logs/experiments.tsv`.
5. Keep only if the metric improved. Then exit.
