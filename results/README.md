Evaluation results committed per stage: `gym/`, `sim/`, `hil/`, `real/`.
`movensys-compare` (`movensys_imitation/compare.py`) regenerates `R1.md`, the plots and the tables in
`plan/plan.md` Section 4 from these.

File naming: `<stage>/<policy>_<variant>_s<seed>.csv`, one row per episode (`episode`, `env_seed`,
`sum_reward`, `max_reward`, `max_coverage`, `success_095`, `success_090`), plus a JSON sidecar of the
same name with the run metadata (train seconds, peak VRAM, latency mean / p95, parameter count,
overrides). `movensys-benchmark` writes both.
