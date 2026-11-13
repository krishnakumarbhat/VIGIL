# Hackathons & GPU projects — archive branch

Snapshot of everything under `gpu /` in the `heaphero2` working tree: hackathon
solutions, notebooks, pipelines, reports, configs and the small datasets needed
to rerun them.

Small files (<= 512 KB) are stored here. Larger payloads stay in the
compressed per-project archives on the external drive; `MANIFEST.tsv` records
every file, its size, sha256 and where it lives.

## Hackathons

- `01-nfl-big-data-bowl-2027/`
- `02-gemma-4-developer-agent-paper/`
- `03-arc-prize-2026-paper-track/`
- `04-arc-prize-2026-arc-agi-2/`
- `05-name-that-llm/`
- `06-playground-s6e10-airline-satisfaction/`
- `07-opencv-ai-competition-2026/`
- `08-openzee-openair-2026/`
- `_gpu_probe/`
- `data/`
- `experiments/`
- `logs/`

## Files

| File | Purpose |
|---|---|
| `MANIFEST.tsv` | every file: `path`, `bytes`, `sha256`, `storage` |
| `restore.sh` | rebuild the original `gpu /` tree |
| `_gpu_root/` | support dirs that sat beside `hackathons/` (`src`, `notebooks`, probes, catboost logs) |

`storage` values: `branch:hackathons/<path>` = already in this branch;
`external:<name>.tar.zst` = inside that archive.

## Restore

```bash
git clone --depth 1 --single-branch --branch hackathons-archive \
  https://github.com/krishnakumarbhat/VIGIL.git vigil-archive

# big files first (each archive is a self-contained subtree, paths preserved)
bash vigil-archive/hackathons/restore.sh \
  /media/pope/9449751218/2026_Solitude_Alone/no_mo_compressed \
  ~/restored/gpu
```

The script untars every archive into the target, then copies the branch files
over the top, so the result matches the original `gpu /` layout.

## Verify

```bash
grep '^hackathons/06-' hackathons/MANIFEST.tsv | head
sha256sum <restored file>       # must equal column 3
```
