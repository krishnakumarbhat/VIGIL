"""Benchmark driver for GG-SPV: graph-guided localization + speculative patch verification.

Runs the ablation matrix over every benchmark task that has local code-graph
artifacts, and emits:

* ``outputs/ablation_results.csv``   - one row per configuration,
* ``outputs/eval_metrics.json``      - aggregates, verifier diagnostics, oracle stats,
* ``outputs/telemetry/<config>.jsonl`` - per-task structured telemetry,
* ``outputs/figures/*.png``          - matplotlib figures from real measurements,
* ``logs/experiments.tsv``           - appended experiment log rows.

The *tools* are deterministic offline re-implementations of the harness tools
driven by the real graphs/embeddings; the *model* is simulated with an explicit
fidelity knob (see ``SimulatedModel``).  Every number written by this script
comes from an actual local execution.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gg_spv_agent import (  # noqa: E402
    AgentConfig,
    AgentLoop,
    CandidatePatch,
    GoldOracle,
    GraphIndex,
    Localizer,
    PatchVerifier,
    TaskTruth,
    index_artifacts,
    load_index_for_task,
    load_tasks,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "gemma4_data", "gemma4_data")
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TEL_DIR = os.path.join(OUT_DIR, "telemetry")
LOG_DIR = os.path.join(ROOT, "logs")

# Declared simulated edit-fidelity knobs per quantized Gemma-4 size.  These are
# assumptions of the *simulated model*, swept in the sensitivity analysis; all
# localization/verification numbers are measured on real artifacts.
MODEL_FIDELITY: Dict[str, float] = {
    "gemma-4-e2b-it-qat-w4a16-ct": 0.45,
    "gemma-4-e4b-it-qat-w4a16-ct": 0.62,
    "gemma-4-12b-it-qat-w4a16-ct": 0.80,
    "gemma-4-31b-it-qat-w4a16-ct": 0.90,
}


def log_experiment(metric: str, value: Any, notes: str) -> None:
    """Append one row to ``logs/experiments.tsv``."""
    path = os.path.join(LOG_DIR, "experiments.tsv")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"02-gg-spv\t{metric}\t{value}\t{notes}\n")


def evaluate_verifier(
    index: GraphIndex,
    truth: TaskTruth,
    localization: Any,
    verifier: PatchVerifier,
    statement: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Measure verifier discrimination on gold vs distractor candidates (real).

    Conditional on localization: only gold edits whose symbol reached the
    localizer's top-20 are scored as positives — the verifier's contract is
    to gate *speculative* patches over the localized neighbourhood, so golds
    outside that neighbourhood are not its errors.
    """
    top20 = set(localization.ranked_symbols[:20])
    out: Dict[str, Any] = {"gold": [], "hard_neg": [], "easy_neg": [], "n_gold_skipped": 0}
    for sym, edit in truth.edits.items():
        if sym not in top20:
            out["n_gold_skipped"] += 1
            continue
        res = verifier.verify(
            CandidatePatch(
                sym, edit.old_text, edit.new_text, "oracle",
                added_text=edit.added_text, removed_text=edit.removed_text,
            ),
            localization,
        )
        out["gold"].append(res.confidence)
        # hard negative: same-file, non-target symbol
        file_prefix = sym.rsplit(".", 1)[0]
        same_file = [
            n
            for n in index.ids
            if n.startswith(file_prefix + ".") and n not in truth.edits
        ]
        for neg in rng.choice(same_file, size=min(3, len(same_file)), replace=False):
            res_n = verifier.verify(
                CandidatePatch(
                    str(neg), edit.old_text, edit.new_text, "distractor",
                    added_text=edit.added_text, removed_text=edit.removed_text,
                ),
                localization,
            )
            out["hard_neg"].append(res_n.confidence)
        # easy negative: random library symbol elsewhere
        lib = list(index.library_set - set(same_file) - set(truth.edits))
        for neg in rng.choice(lib, size=min(3, len(lib)), replace=False):
            res_n = verifier.verify(
                CandidatePatch(
                    str(neg), edit.old_text, edit.new_text, "distractor",
                    added_text=edit.added_text, removed_text=edit.removed_text,
                ),
                localization,
            )
            out["easy_neg"].append(res_n.confidence)
    return out


def auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """Rank-based AUROC (ties count as half)."""
    if not scores_pos or not scores_neg:
        return float("nan")
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    diff = (pos[:, None] > neg[None, :]).mean() + 0.5 * (
        pos[:, None] == neg[None, :]
    ).mean()
    return float(diff)


def run_config(
    name: str,
    tasks: List[Dict[str, Any]],
    artifacts: Dict[str, Dict[str, str]],
    config: AgentConfig,
    strategy: str,
    use_verifier: bool,
    hops: Optional[int],
    keep_verifier_stats: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Run one ablation configuration over all covered tasks."""
    loop_cache: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    verifier_scores: Dict[str, List[float]] = {
        "gold": [],
        "hard_neg": [],
        "easy_neg": [],
    }
    attribution: Dict[str, int] = {"symbol": 0, "file": 0}
    start = time.perf_counter()
    for task in tasks:
        commit = task["base_commit"]
        if commit not in loop_cache:
            index = load_index_for_task(task, artifacts)
            if index is None:
                continue
            loop_cache[commit] = (index, AgentLoop(index, config, use_verifier=use_verifier, hops=hops))
        index, loop = loop_cache[commit]
        truth = GoldOracle.build(task, index.nodes)
        attribution[truth.gt_mode] += 1
        outcome = loop.run_task(task, truth, strategy=strategy, config_name=name)
        rows.append(dataclasses.asdict(outcome) | {"telemetry": outcome.telemetry})
        if keep_verifier_stats and truth.edits:
            loc = Localizer(index, loop.config)
            localization = (
                loc.localize_gg(task["problem_statement"])
                if strategy == "gg"
                else loc.localize_flat(task["problem_statement"])
            )
            rng = np.random.default_rng(abs(hash(task["instance_id"])) % (2**32))
            stats = evaluate_verifier(
                index, truth, localization, PatchVerifier(index, loop.config),
                task["problem_statement"], rng,
            )
            for key, values in stats.items():
                if key == "n_gold_skipped":
                    verifier_scores.setdefault("n_gold_skipped", []).append(values)
                else:
                    verifier_scores[key].extend(values)
    wall = time.perf_counter() - start
    symbol_tasks = [r for r in rows if r["telemetry"]["gt_mode"] == "symbol"]
    n = max(1, len(rows))
    nsym = max(1, len(symbol_tasks))
    recall = {
        k: sum(1 for r in symbol_tasks if r["recall_at"][k]) / nsym for k in ("1", "3", "5")
    }
    # file-level recall: any of the top-5 symbols belongs to a gold-patched file
    file_hits = 0
    for r in rows:
        gold_files = set(r["telemetry"].get("gold_files", []))
        top_files = {
            s.rsplit(".", 1)[0].replace(".", "/") + ".py"
            for s in r["telemetry"]["top5"]
        }
        if gold_files and top_files & gold_files:
            file_hits += 1
    summary: Dict[str, Any] = {
        "configuration": name,
        "model_id": config.model_id,
        "simulated_model": config.simulated,
        "fidelity": config.fidelity,
        "strategy": strategy,
        "hops": hops,
        "use_verifier": use_verifier,
        "n_tasks": len(rows),
        "n_symbol_gt": len(symbol_tasks),
        "recall_at_1": round(recall["1"], 4),
        "recall_at_3": round(recall["3"], 4),
        "recall_at_5": round(recall["5"], 4),
        "file_recall_at_5": round(file_hits / n, 4),
        "mean_tokens_read": round(statistics.mean(r["tokens_read"] for r in rows), 1) if rows else 0.0,
        "median_tokens_read": round(statistics.median(r["tokens_read"] for r in rows), 1) if rows else 0.0,
        "mean_tokens_to_recall_5": round(
            statistics.mean(
                r["telemetry"]["tokens_to_rank"].get("5", r["tokens_read"]) for r in symbol_tasks
            ),
            1,
        ) if symbol_tasks else 0.0,
        "patch_pass_rate": round(sum(1 for r in rows if r["patch_correct"]) / n, 4),
        "verifier_accept_rate": round(sum(1 for r in rows if r["verifier_accepted"]) / n, 4),
        "mean_tool_calls": round(statistics.mean(r["tool_calls"] for r in rows), 2) if rows else 0.0,
        "wall_time_s": round(wall, 2),
    }
    extra: Dict[str, Any] = {"attribution": attribution}
    if verifier_scores["gold"]:
        gold = verifier_scores["gold"]
        neg_h = verifier_scores["hard_neg"]
        neg_e = verifier_scores["easy_neg"]
        thr = config.accept_threshold
        extra["verifier"] = {
            "n_gold_localized": len(gold),
            "n_gold_skipped_unlocalized": sum(verifier_scores.get("n_gold_skipped", [])),
            "n_hard_neg": len(neg_h),
            "n_easy_neg": len(neg_e),
            "accept_threshold": thr,
            "tpr_at_threshold": round(sum(1 for g in gold if g >= thr) / len(gold), 4),
            "fpr_hard_at_threshold": round(sum(1 for g in neg_h if g >= thr) / max(1, len(neg_h)), 4),
            "fpr_easy_at_threshold": round(sum(1 for g in neg_e if g >= thr) / max(1, len(neg_e)), 4),
            "gold_mean_conf": round(statistics.mean(gold), 4),
            "hard_neg_mean_conf": round(statistics.mean(neg_h), 4),
            "easy_neg_mean_conf": round(statistics.mean(neg_e), 4),
            "auroc_hard": round(auroc(gold, neg_h), 4),
            "auroc_easy": round(auroc(gold, neg_e), 4),
            "scores": {k: v for k, v in verifier_scores.items() if k != "n_gold_skipped"},
        }
    return summary, rows, extra


def make_figures(summaries: List[Dict[str, Any]], verifier: Dict[str, Any]) -> None:
    """Publication figures from the measured ablation numbers."""
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})

    # Fig 1: recall@5 vs mean tokens read (localization efficiency frontier).
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    styles = {
        "flat_baseline": ("s", "tab:red", "flat read (no graph)"),
        "gg_hops0": ("^", "tab:orange", "GG, seeds only"),
        "gg_hops1": ("D", "tab:green", "GG, 1-hop"),
        "gg_spv_12b": ("o", "tab:blue", "GG, 2-hop (full GG-SPV)"),
    }
    for s in summaries:
        key = s["configuration"]
        if key in styles:
            marker, color, label = styles[key]
            ax.scatter(s["mean_tokens_read"], s["recall_at_5"], marker=marker,
                       color=color, s=70, label=label)
            ax.annotate(key, (s["mean_tokens_read"], s["recall_at_5"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("mean tokens read per task (chars/4 proxy, log scale)")
    ax.set_ylabel("localization recall@5 (symbol ground truth)")
    ax.set_title("Graph-guided vs flat localization: accuracy vs context cost")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_recall_vs_tokens.png"))
    plt.close(fig)

    # Fig 2: ablation bars - recall@k by hops, plus token cost.
    ablation = [s for s in summaries if s["configuration"] in
                ("flat_baseline", "gg_hops0", "gg_hops1", "gg_spv_12b")]
    labels = [s["configuration"] for s in ablation]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    width = 0.25
    for i, (k, color) in enumerate((("recall_at_1", "tab:blue"),
                                    ("recall_at_3", "tab:green"),
                                    ("recall_at_5", "tab:orange"))):
        ax.bar(x + (i - 1) * width, [s[k] for s in ablation], width, label=k, color=color)
    ax2 = ax.twinx()
    ax2.plot(x, [s["mean_tokens_read"] for s in ablation], "k.-", label="mean tokens")
    ax2.set_ylabel("mean tokens read")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("recall")
    ax.set_title("Ablation: expansion hops and token cost")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_ablation_bars.png"))
    plt.close(fig)

    # Fig 3: per-model (simulated) patch pass rate with/without verification.
    model_rows = [s for s in summaries if s["configuration"].startswith(("model_", "gg_spv_"))]
    seen: Dict[str, Dict[str, float]] = {}
    for s in model_rows:
        seen.setdefault(s["model_id"], {})["verifier" if s["use_verifier"] else "no_verifier"] = s["patch_pass_rate"]
    models = [m for m in MODEL_FIDELITY if m in seen]
    short = [m.split("-")[1] for m in models]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.arange(len(models))
    ax.bar(xs - 0.18, [seen[m].get("no_verifier", 0) for m in models], 0.36,
           label="no verification", color="tab:red")
    ax.bar(xs + 0.18, [seen[m].get("verifier", 0) for m in models], 0.36,
           label="GG-SPV verification gate", color="tab:green")
    for i, m in enumerate(models):
        ax.text(i, 0.02, f"f={MODEL_FIDELITY[m]}", fontsize=7, ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(short)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("patch pass rate (simulated model)")
    ax.set_title("Simulated model-scale ablation (fidelity knob f)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_model_ablation.png"))
    plt.close(fig)

    # Fig 4: verifier score distributions (real candidates).
    if verifier.get("scores"):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for key, color, label in (
            ("gold", "tab:green", "gold-derived edits"),
            ("hard_neg", "tab:orange", "same-file distractors"),
            ("easy_neg", "tab:red", "cross-file distractors"),
        ):
            vals = verifier["scores"][key]
            if vals:
                ax.hist(vals, bins=20, alpha=0.55, color=color, label=f"{label} (n={len(vals)})")
        ax.axvline(0.5, color="k", linestyle="--", linewidth=1, label="accept threshold 0.5")
        ax.set_xlabel("verifier confidence")
        ax.set_ylabel("count")
        ax.set_title("Speculative patch verification: confidence distributions")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig4_verifier_distribution.png"))
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="GG-SPV benchmark runner")
    parser.add_argument("--limit", type=int, default=0, help="cap number of tasks (0 = all)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    tasks = load_tasks(os.path.join(DATA_DIR, "tasks.jsonl"))
    artifacts = index_artifacts(DATA_DIR)
    covered = [
        t for t in tasks
        if t["base_commit"] in artifacts and len(artifacts[t["base_commit"]]) == 2
    ]
    covered.sort(key=lambda t: t["instance_id"])
    if args.limit:
        covered = covered[: args.limit]
    print(f"benchmark: {len(covered)} tasks with local graph+embedding artifacts "
          f"(of {len(tasks)} total)")

    base = AgentConfig()
    configs: List[Tuple[str, AgentConfig, str, bool, Optional[int]]] = [
        # name, config, strategy, use_verifier, hops
        ("flat_baseline", dataclasses.replace(base, model_id="gemma-4-12b-it-qat-w4a16-ct",
                                              fidelity=MODEL_FIDELITY["gemma-4-12b-it-qat-w4a16-ct"]),
         "flat", False, None),
        ("gg_hops0", dataclasses.replace(base, model_id="gemma-4-12b-it-qat-w4a16-ct", fidelity=MODEL_FIDELITY["gemma-4-12b-it-qat-w4a16-ct"]),
         "gg", False, 0),
        ("gg_hops1", dataclasses.replace(base, model_id="gemma-4-12b-it-qat-w4a16-ct", fidelity=MODEL_FIDELITY["gemma-4-12b-it-qat-w4a16-ct"]),
         "gg", False, 1),
        ("gg_spv_noverify", dataclasses.replace(base, model_id="gemma-4-12b-it-qat-w4a16-ct", fidelity=MODEL_FIDELITY["gemma-4-12b-it-qat-w4a16-ct"]),
         "gg", False, 2),
    ]
    for model_id, fidelity in MODEL_FIDELITY.items():
        suffix = model_id.split("-")[2]
        configs.append((
            f"gg_spv_{suffix}",
            dataclasses.replace(base, model_id=model_id, fidelity=fidelity),
            "gg", True, 2,
        ))
    # Sensitivity sweep on the fidelity knob (12b scaffold).
    for f in (0.30, 0.50, 0.70, 1.00):
        configs.append((
            f"sensitivity_f{int(f * 100):02d}",
            dataclasses.replace(base, model_id="gemma-4-12b-it-qat-w4a16-ct", fidelity=f),
            "gg", True, 2,
        ))

    summaries: List[Dict[str, Any]] = []
    verifier_extra: Dict[str, Any] = {}
    for name, cfg, strategy, use_verifier, hops in configs:
        t0 = time.time()
        keep = name == "gg_spv_12b"
        summary, rows, extra = run_config(
            name, covered, artifacts, cfg, strategy, use_verifier, hops,
            keep_verifier_stats=keep,
        )
        summaries.append(summary)
        with open(os.path.join(TEL_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        if keep and "verifier" in extra:
            verifier_extra = extra["verifier"]
        print(f"[{name}] recall@1/3/5 = {summary['recall_at_1']:.3f}/"
              f"{summary['recall_at_3']:.3f}/{summary['recall_at_5']:.3f}  "
              f"tokens={summary['mean_tokens_read']:.0f}  "
              f"pass={summary['patch_pass_rate']:.3f}  "
              f"({time.time() - t0:.1f}s)")

    # CSV -------------------------------------------------------------------
    csv_path = os.path.join(OUT_DIR, "ablation_results.csv")
    fieldnames = list(summaries[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    # JSON metrics ----------------------------------------------------------
    metrics = {
        "benchmark": {
            "tasks_total": len(tasks),
            "tasks_covered": len(covered),
            "repos": sorted({t["repo"] for t in covered}),
            "token_proxy": "chars/4 (offline proxy for the harness token accounting)",
            "simulated_model_note": (
                "Patch generation is simulated: candidates are materialized from "
                "gold patches with a declared per-model fidelity knob f; all "
                "localization and verification numbers are measured on real "
                "competition artifacts."
            ),
        },
        "configurations": summaries,
        "verifier_diagnostics": verifier_extra,
    }
    with open(os.path.join(OUT_DIR, "eval_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    if verifier_extra:
        make_figures(summaries, verifier_extra)
    else:
        make_figures(summaries, {})

    # experiments.tsv -------------------------------------------------------
    by_name = {s["configuration"]: s for s in summaries}
    log_experiment("n_tasks_covered", len(covered), "graph+embedding artifacts available locally")
    for name in ("flat_baseline", "gg_hops0", "gg_hops1", "gg_spv_12b"):
        s = by_name.get(name)
        if s:
            log_experiment(f"recall_at_5/{name}", s["recall_at_5"], "symbol ground truth")
            log_experiment(f"recall_at_1/{name}", s["recall_at_1"], "symbol ground truth")
            log_experiment(f"mean_tokens_read/{name}", s["mean_tokens_read"], "chars/4 proxy")
    if verifier_extra:
        log_experiment("verifier_auroc_hard_neg", verifier_extra["auroc_hard"], "gold vs same-file distractors")
        log_experiment("verifier_gold_mean_conf", verifier_extra["gold_mean_conf"], "accept threshold 0.5")
    print(f"wrote {csv_path}")
    print(f"wrote {os.path.join(OUT_DIR, 'eval_metrics.json')}")
    print(f"figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
