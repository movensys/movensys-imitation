"""Plan task P1.5: tables and plots from the evaluation CSVs (``movensys-compare``).

Reads ``results/<stage>/<policy>_<variant>_s<seed>.csv`` (one row per episode, written by
``movensys-benchmark``) and the JSON sidecars, aggregates over seeds (mean and sample standard
deviation) and writes

* ``results/<stage>/R1.md``: the table in the plan's Section 4 format,
* ``results/<stage>/r1_success_vs_steps.png``: success rate against inference steps,
* optionally (``--write-plan``) the same rows into ``plan/plan.md`` Table R1.

    movensys-compare [--stage gym] [--write-plan]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLAN = ROOT / "plan" / "plan.md"

log = logging.getLogger("compare")

# Row order and labels of Table R1 (plan Section 4). (policy, variant) -> label.
R1_ROWS: tuple[tuple[tuple[str, str], str], ...] = (
    (("diffusion", "ref200k"), "Diffusion, 200 k reference (model card: 65.4 %)"),
    (("act_default", "default"), "ACT default (chunk 100, ensembling)"),
    (("act_default_novae", "default"), "ACT default, no VAE"),
    (("act_matched", "default"), "ACT matched 16/8"),
    (("act_matched_novae", "default"), "ACT matched, no VAE"),
    (("diffusion", "ddim10"), "Diffusion DDIM 10"),
    (("diffusion", "ddpm100"), "Diffusion DDPM 100"),
    (("flow_matching", "euler1"), "Flow Matching Euler 1"),
    (("flow_matching", "euler2"), "Flow Matching Euler 2"),
    (("flow_matching", "euler4"), "Flow Matching Euler 4"),
    (("flow_matching", "euler8"), "Flow Matching Euler 8"),
    (("flow_matching", "euler16"), "Flow Matching Euler 16"),
)
R1_HEADER = (
    "| Policy / variant | Success 0.95 | Success 0.90 | Avg max cov | Train h | VRAM GB "
    "| Latency ms (p95) |\n"
    "|---|---|---|---|---|---|---|\n"
)
_FILE_RE = re.compile(r"^(?P<stem>.+)_s(?P<seed>\d+)\.csv$")


# ----------------------------------------------------------------------------- loading


def parse_name(path: Path) -> tuple[str, str, int]:
    """``act_matched_novae_default_s2.csv`` -> ("act_matched_novae", "default", 2)."""
    m = _FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"unexpected result file name {path.name}")
    policy, variant = m["stem"].rsplit("_", 1)
    return policy, variant, int(m["seed"])


def load_run(csv_path: Path) -> dict:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    meta_path = csv_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n = len(rows)
    return {
        "n_episodes": n,
        "success_095": 100.0 * sum(int(r["success_095"]) for r in rows) / n,
        "success_090": 100.0 * sum(int(r["success_090"]) for r in rows) / n,
        "avg_max_coverage": sum(float(r["max_coverage"]) for r in rows) / n,
        "train_h": meta["train_s"] / 3600 if meta.get("train_s") else None,
        "vram_gb": meta.get("peak_vram_gb"),
        "latency_p95": meta.get("latency_ms_p95"),
        "latency_mean": meta.get("latency_ms_mean"),
    }


def load_stage(stage_dir: Path) -> dict[tuple[str, str], dict[int, dict]]:
    runs: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for csv_path in sorted(stage_dir.glob("*.csv")):
        policy, variant, seed = parse_name(csv_path)
        runs[(policy, variant)][seed] = load_run(csv_path)
    return runs


# ----------------------------------------------------------------------------- aggregation


def mean_std(values: list[float | None]) -> tuple[float, float, int] | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def aggregate(seed_runs: dict[int, dict]) -> dict[str, tuple[float, float, int] | None]:
    keys = (
        "success_095",
        "success_090",
        "avg_max_coverage",
        "train_h",
        "vram_gb",
        "latency_p95",
        "latency_mean",
    )
    return {k: mean_std([r.get(k) for r in seed_runs.values()]) for k in keys}


def fmt(agg: tuple[float, float, int] | None, digits: int = 1, pm: bool = True) -> str:
    if agg is None:
        return ""
    mean, std, n = agg
    if n > 1 and pm:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def r1_rows(runs: dict[tuple[str, str], dict[int, dict]]) -> list[str]:
    lines = []
    for key, label in R1_ROWS:
        if key not in runs:
            lines.append(f"| {label} | | | | | | |")
            continue
        a = aggregate(runs[key])
        n = len(runs[key])
        lat = fmt(a["latency_p95"], 1, pm=False)
        lines.append(
            f"| {label} | {fmt(a['success_095'])} | {fmt(a['success_090'])} | "
            f"{fmt(a['avg_max_coverage'], 2)} | {fmt(a['train_h'], 1, pm=False)} | "
            f"{fmt(a['vram_gb'], 1, pm=False)} | {lat} |"
            + (f" <!-- {n} seed(s) -->" if n != 3 else "")
        )
    return lines


def write_r1_md(stage_dir: Path, runs) -> Path:
    n_ep = {r["n_episodes"] for seed_runs in runs.values() for r in seed_runs.values()}
    text = (
        f"# R1. gym-pusht\n\nMean ± sample std over training seeds; "
        f"{', '.join(map(str, sorted(n_ep)))} fixed episodes per run (seeds 100000+). "
        "Success = max coverage ≥ 0.95 (secondary 0.90). Latency: one model call, batch 1, "
        "pre/post-processing excluded, p95 over 100 calls.\n\n"
        + R1_HEADER
        + "\n".join(r1_rows(runs))
        + "\n"
    )
    path = stage_dir / "R1.md"
    path.write_text(text)
    return path


def write_plan_table(runs) -> None:
    """Replace the body of Table R1 in plan/plan.md (rows between the header and the blank line)."""
    text = PLAN.read_text()
    start = text.index("**R1. gym-pusht**")
    header_end = text.index("|---|---|---|---|---|---|---|\n", start) + len(
        "|---|---|---|---|---|---|---|\n"
    )
    body_end = text.index("\n\n", header_end)
    new = text[:header_end] + "\n".join(r1_rows(runs)) + text[body_end:]
    PLAN.write_text(new)
    log.info("updated Table R1 in %s", PLAN)


# ----------------------------------------------------------------------------- plot


def plot_success_vs_steps(stage_dir: Path, runs) -> Path | None:
    """Success (0.95) against inference steps: Flow Matching Euler K, Diffusion DDIM / DDPM,
    ACT matched as a reference line. One axis, fixed series colours, direct labels."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        log.warning("matplotlib not installed; skipping the plot")
        return None

    series = {  # fixed categorical order: blue, orange, aqua (dataviz palette)
        "Flow Matching (Euler)": (
            "#2a78d6",
            [(k, ("flow_matching", f"euler{k}")) for k in (1, 2, 4, 8, 16)],
        ),
        "Diffusion (DDIM 10, DDPM 100)": (
            "#eb6834",
            [(10, ("diffusion", "ddim10")), (100, ("diffusion", "ddpm100"))],
        ),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    plotted = False
    for label, (color, points) in series.items():
        xs, ys, es = [], [], []
        for x, key in points:
            if key in runs:
                a = aggregate(runs[key])["success_095"]
                xs.append(x)
                ys.append(a[0])
                es.append(a[1])
        if not xs:
            continue
        plotted = True
        ax.errorbar(xs, ys, yerr=es, color=color, marker="o", ms=5, lw=2, capsize=3, label=label)
        ax.annotate(
            label,
            (xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color="#52514e",
            fontsize=8,
            va="center",
        )
    for key, label, color in (
        (("act_matched", "default"), "ACT matched 16/8", "#1baf7a"),
        (("diffusion", "ref200k"), "Diffusion 200 k reference", "#52514e"),
    ):
        if key in runs:
            y = aggregate(runs[key])["success_095"][0]
            ax.axhline(y, color=color, lw=1.5, ls="--")
            ax.annotate(
                label,
                (1, y),
                xytext=(2, 3),
                textcoords="offset points",
                color="#52514e",
                fontsize=8,
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 100])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "100"])
    ax.set_xlabel("inference steps (model calls per chunk)")
    ax.set_ylabel("success rate, coverage ≥ 0.95 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = stage_dir / "r1_success_vs_steps.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", default="gym")
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument(
        "--write-plan", action="store_true", help="also rewrite Table R1 in plan/plan.md"
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    stage_dir = Path(args.results_dir) / args.stage
    runs = load_stage(stage_dir)
    if not runs:
        raise SystemExit(f"no result CSVs in {stage_dir}")
    for key, seed_runs in sorted(runs.items()):
        a = aggregate(seed_runs)
        log.info(
            "%-22s %-9s seeds=%d success95=%s success90=%s",
            *key,
            len(seed_runs),
            fmt(a["success_095"]),
            fmt(a["success_090"]),
        )
    print(R1_HEADER + "\n".join(r1_rows(runs)))
    log.info("wrote %s", write_r1_md(stage_dir, runs))
    if not args.no_plot:
        plot = plot_success_vs_steps(stage_dir, runs)
        if plot:
            log.info("wrote %s", plot)
    if args.write_plan:
        write_plan_table(runs)


if __name__ == "__main__":
    main()
