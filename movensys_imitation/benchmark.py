"""Plan task P1.4: the gym-pusht benchmark driver (``movensys-benchmark``).

Runs the training matrix (ACT x 4, Diffusion, Flow Matching) over the seeds, evaluates every
inference variant of each checkpoint on the same 50 fixed episodes, measures model latency, and
writes one CSV per (policy, variant, seed) to ``results/gym/`` in the format ``compare.py``
reads (one row per episode) plus a JSON sidecar with train time, peak VRAM and latency.

Every step is idempotent: finished trainings and evaluations are skipped, partial trainings are
resumed, so the command can be re-run after an interruption.

    movensys-benchmark run [--only act_matched,flow_matching] [--seeds 1,2,3] [--steps 100000]
    movensys-benchmark import-eval outputs/ref_diffusion_pusht/eval_final/eval_info.json \
        --name diffusion --variant ref200k --seed 100000 --train-s 16213 --vram-gb 4.98 \
        --pretrained outputs/ref_diffusion_pusht/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / "configs"
RESULTS = ROOT / "results" / "gym"
OUT_ROOT = ROOT / "outputs" / "gym"

# gym-pusht: reward = clip(coverage / 0.95, 0, 1); success = coverage > 0.95 (plan Section 4).
SUCCESS_THRESHOLD = 0.95
SECONDARY_THRESHOLD = 0.90
EVAL_SEED = 100_000  # fixed episode seeds EVAL_SEED .. EVAL_SEED + n_eval - 1 for every run
DEFAULT_SEEDS = (1, 2, 3)
TASK = "Push the T-shaped block onto the T-shaped target."

log = logging.getLogger("benchmark")


@dataclass(frozen=True)
class Variant:
    """An inference-time setting evaluated on the same checkpoint."""

    name: str
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class Training:
    """One training configuration = policy YAML + CLI overrides."""

    name: str  # output-dir and CSV prefix; must not contain the variant separator pattern
    policy_yaml: str
    overrides: tuple[str, ...] = ()
    variants: tuple[Variant, ...] = (Variant("default"),)


_ACT_MATCHED = (
    "--policy.chunk_size=16",
    "--policy.n_action_steps=8",
    "--policy.temporal_ensemble_coeff=null",
)

MATRIX: tuple[Training, ...] = (
    Training(
        "diffusion",
        "diffusion",
        variants=(
            Variant(
                "ddim10",
                ("--policy.noise_scheduler_type=DDIM", "--policy.num_inference_steps=10"),
            ),
            Variant(
                "ddpm100",
                ("--policy.noise_scheduler_type=DDPM", "--policy.num_inference_steps=100"),
            ),
        ),
    ),
    Training(
        "flow_matching",
        "flow_matching",
        variants=tuple(
            Variant(f"euler{k}", (f"--policy.fm_num_steps={k}",)) for k in (1, 2, 4, 8, 16)
        ),
    ),
    Training("act_default", "act"),
    Training("act_default_novae", "act", ("--policy.use_vae=false",)),
    Training("act_matched", "act", _ACT_MATCHED),
    Training("act_matched_novae", "act", (*_ACT_MATCHED, "--policy.use_vae=false")),
)


# ----------------------------------------------------------------------------- subprocesses


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("MUJOCO_GL", "egl")
    return env


def _run(cmd: list[str], log_path: Path, dry_run: bool) -> float:
    """Run ``cmd`` appending stdout+stderr to ``log_path``; return wall seconds."""
    log.info("$ %s", " ".join(cmd))
    if dry_run:
        return 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=_env(), cwd=ROOT)
    return time.perf_counter() - t0


def _cli(command: str) -> list[str]:
    return [sys.executable, "-m", f"movensys_imitation.cli.{command}"]


# ----------------------------------------------------------------------------- training


def train_dir(out_root: Path, training: Training, seed: int) -> Path:
    return out_root / f"{training.name}_s{seed}"


def _last_checkpoint_step(out: Path) -> int | None:
    """Step of the ``last`` checkpoint from LeRobot's ``training_state/training_step.json``."""
    step_file = out / "checkpoints" / "last" / "training_state" / "training_step.json"
    if not step_file.exists():
        return None
    try:
        return int(json.loads(step_file.read_text())["step"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def pretrained_dir(out: Path) -> Path:
    return out / "checkpoints" / "last" / "pretrained_model"


def train_log_path(out: Path) -> Path:
    """Next to, not inside, the output dir: LeRobot refuses an existing output_dir."""
    return out.parent / f"{out.name}.train.log"


def peak_vram_gb(train_log: Path) -> float | None:
    """Max of LeRobot's ``mem_gb`` (``torch.cuda.max_memory_allocated``) over the train log."""
    if not train_log.exists():
        return None
    values = [
        float(m) for m in re.findall(r"mem_gb:([0-9.]+)", train_log.read_text(errors="ignore"))
    ]
    return max(values) if values else None


def ensure_trained(
    training: Training, seed: int, steps: int, out_root: Path, dry_run: bool
) -> dict:
    out = train_dir(out_root, training, seed)
    meta_path = out / "train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"train_s": 0.0}
    done = _last_checkpoint_step(out)
    if done is not None and done >= steps:
        log.info("[%s s%d] trained (%d steps), skipping", training.name, seed, done)
    else:
        if done is not None:
            log.info("[%s s%d] resuming from step %d", training.name, seed, done)
            cmd = _cli("train") + [
                f"--config_path={pretrained_dir(out) / 'train_config.json'}",
                "--resume=true",
            ]
        else:
            if out.exists() and not dry_run:
                shutil.rmtree(out)  # LeRobot refuses an existing output_dir without a checkpoint
            cmd = _cli("train") + [
                f"--config_path={CONFIGS / 'pusht_gym.yaml'},{CONFIGS / training.policy_yaml}.yaml",
                f"--seed={seed}",
                f"--steps={steps}",
                f"--output_dir={out}",
                *training.overrides,
            ]
        meta["train_s"] = meta.get("train_s", 0.0) + _run(cmd, train_log_path(out), dry_run)
        if not dry_run:
            meta_path.write_text(json.dumps(meta, indent=2))
    meta["peak_vram_gb"] = peak_vram_gb(train_log_path(out))
    meta["steps"] = steps
    return meta


# ----------------------------------------------------------------------------- evaluation


def eval_cmd(pretrained: Path, out_dir: Path, n_eval: int, overrides: tuple[str, ...]) -> list[str]:
    return _cli("eval") + [
        f"--policy.path={pretrained}",
        "--env.type=pusht",
        f"--eval.n_episodes={n_eval}",
        f"--eval.batch_size={n_eval}",
        "--eval.use_async_envs=false",
        f"--seed={EVAL_SEED}",
        f"--output_dir={out_dir}",
        *overrides,
    ]


def ensure_evaluated(
    out: Path, variant: Variant, n_eval: int, dry_run: bool, log_name: str
) -> Path | None:
    eval_dir = out / f"eval_{variant.name}"
    info = eval_dir / "eval_info.json"
    if info.exists():
        log.info("[%s] eval %s exists, skipping", log_name, variant.name)
        return info
    if eval_dir.exists() and not dry_run:
        shutil.rmtree(eval_dir)  # half-finished eval
    _run(
        eval_cmd(pretrained_dir(out), eval_dir, n_eval, variant.overrides),
        out / "eval.log",
        dry_run,
    )
    return info if info.exists() else None


def episodes_from_eval_info(info_path: Path) -> list[dict]:
    """One row per episode from LeRobot's ``eval_info.json`` (rewards are coverage / 0.95)."""
    info = json.loads(info_path.read_text())
    metrics = info["per_task"][0]["metrics"]
    rows = []
    for i, (sum_r, max_r, succ) in enumerate(
        zip(metrics["sum_rewards"], metrics["max_rewards"], metrics["successes"], strict=True)
    ):
        max_cov = max_r * SUCCESS_THRESHOLD
        rows.append(
            {
                "episode": i,
                "env_seed": EVAL_SEED + i,
                "sum_reward": round(sum_r, 4),
                "max_reward": round(max_r, 6),
                "max_coverage": round(max_cov, 6),
                "success_095": int(bool(succ)),
                "success_090": int(max_cov >= SECONDARY_THRESHOLD - 1e-9),
            }
        )
    return rows


def write_result(
    results_dir: Path, name: str, variant: str, seed: int, rows: list[dict], meta: dict
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = results_dir / f"{name}_{variant}_s{seed}"
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return stem.with_suffix(".csv")


# ----------------------------------------------------------------------------- latency


def measure_latency(
    pretrained: Path, overrides: tuple[str, ...], n: int = 100, warmup: int = 10
) -> dict:
    """Time one ``select_action`` call after ``reset()`` (= one model inference: a full action
    chunk for chunked policies, one ensembled step for ACT with temporal ensembling).

    Batch of one, inputs already on the GPU, pre/post-processing excluded. Returns mean and p95
    in milliseconds and the parameter count.
    """
    import numpy as np
    import torch
    from lerobot.configs import FeatureType, PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    from movensys_imitation.policies import register_all

    register_all()
    # PreTrainedConfig.from_pretrained takes policy-level args (LeRobot's CLI strips "policy.").
    cfg_args = [arg.replace("--policy.", "--", 1) for arg in overrides]
    cfg = PreTrainedConfig.from_pretrained(str(pretrained), cli_overrides=cfg_args)
    for arg in cfg_args:  # guard against a silently ignored override
        key, value = arg.removeprefix("--").split("=", 1)
        actual = getattr(cfg, key)
        if str(actual).lower() != value.lower() and str(actual) != value:
            try:
                assert float(actual) == float(value)
            except (TypeError, ValueError, AssertionError) as exc:
                raise RuntimeError(f"override {arg} not applied: {key}={actual!r}") from exc
    cfg.pretrained_path = pretrained
    policy = get_policy_class(cfg.type).from_pretrained(str(pretrained), config=cfg)
    policy = policy.to(cfg.device).eval()
    device = torch.device(cfg.device)
    obs = {}
    for key, ft in cfg.input_features.items():
        if ft.type is FeatureType.VISUAL:
            obs[key] = torch.rand(1, *ft.shape, device=device)
        else:
            obs[key] = torch.rand(1, *ft.shape, device=device) * 2 - 1

    def one_call() -> float:
        policy.reset()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            policy.select_action(dict(obs))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        one_call()
    times = np.array([one_call() for _ in range(n)])
    return {
        "latency_ms_mean": float(times.mean()),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "latency_n": n,
        "n_params": sum(p.numel() for p in policy.parameters()),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }


# ----------------------------------------------------------------------------- commands


def run_matrix(args: argparse.Namespace) -> None:
    only = set(args.only.split(",")) if args.only else None
    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out_root)
    results_dir = Path(args.results_dir)
    for training in MATRIX:
        if only and training.name not in only:
            continue
        for seed in seeds:
            tag = f"{training.name} s{seed}"
            meta = ensure_trained(training, seed, args.steps, out_root, args.dry_run)
            out = train_dir(out_root, training, seed)
            for variant in training.variants:
                csv_path = results_dir / f"{training.name}_{variant.name}_s{seed}.csv"
                if csv_path.exists() and not args.redo_results:
                    log.info("[%s] %s: result exists, skipping", tag, variant.name)
                    continue
                info = ensure_evaluated(out, variant, args.n_eval, args.dry_run, tag)
                if args.dry_run or info is None:
                    continue
                run_meta = {
                    "policy": training.name,
                    "policy_yaml": training.policy_yaml,
                    "variant": variant.name,
                    "seed": seed,
                    "train_overrides": list(training.overrides),
                    "eval_overrides": list(variant.overrides),
                    "n_eval": args.n_eval,
                    "eval_seed": EVAL_SEED,
                    **meta,
                }
                if not args.skip_latency:
                    run_meta.update(measure_latency(pretrained_dir(out), variant.overrides))
                path = write_result(
                    results_dir,
                    training.name,
                    variant.name,
                    seed,
                    episodes_from_eval_info(info),
                    run_meta,
                )
                log.info(
                    "[%s] wrote %s",
                    tag,
                    path.relative_to(ROOT) if path.is_relative_to(ROOT) else path,
                )


def import_eval(args: argparse.Namespace) -> None:
    rows = episodes_from_eval_info(Path(args.eval_info))
    meta = {
        "policy": args.name,
        "variant": args.variant,
        "seed": args.seed,
        "train_s": args.train_s,
        "peak_vram_gb": args.vram_gb,
        "steps": args.steps,
        "n_eval": len(rows),
        "eval_seed": EVAL_SEED,
        "imported_from": str(args.eval_info),
    }
    if args.pretrained and not args.skip_latency:
        meta.update(measure_latency(Path(args.pretrained), tuple(args.override or ())))
    path = write_result(Path(args.results_dir), args.name, args.variant, args.seed, rows, meta)
    log.info("wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="train, evaluate and record the benchmark matrix")
    p_run.add_argument("--only", help="comma-separated training names (default: all)")
    p_run.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p_run.add_argument("--steps", type=int, default=100_000)
    p_run.add_argument("--n-eval", type=int, default=50)
    p_run.add_argument("--out-root", default=str(OUT_ROOT))
    p_run.add_argument("--results-dir", default=str(RESULTS))
    p_run.add_argument("--dry-run", action="store_true", help="print the commands only")
    p_run.add_argument("--skip-latency", action="store_true")
    p_run.add_argument(
        "--redo-results", action="store_true", help="rewrite CSVs from existing evals"
    )
    p_run.add_argument("--list", action="store_true", help="list the matrix and exit")
    p_run.set_defaults(func=run_matrix)

    p_imp = sub.add_parser("import-eval", help="convert a LeRobot eval_info.json into a result CSV")
    p_imp.add_argument("eval_info")
    p_imp.add_argument("--name", required=True)
    p_imp.add_argument("--variant", required=True)
    p_imp.add_argument("--seed", type=int, required=True)
    p_imp.add_argument("--train-s", type=float)
    p_imp.add_argument("--vram-gb", type=float)
    p_imp.add_argument("--steps", type=int)
    p_imp.add_argument("--pretrained", help="checkpoint dir; enables the latency measurement")
    p_imp.add_argument("--override", action="append", help="policy override for the latency run")
    p_imp.add_argument("--skip-latency", action="store_true")
    p_imp.add_argument("--results-dir", default=str(RESULTS))
    p_imp.set_defaults(func=import_eval)

    args = parser.parse_args(argv)
    if getattr(args, "list", False):
        for t in MATRIX:
            print(f"{t.name:20s} {t.policy_yaml:14s} {' '.join(t.overrides)}")
            for v in t.variants:
                print(f"    {v.name:10s} {' '.join(v.overrides)}")
        return
    args.func(args)


if __name__ == "__main__":
    main()
