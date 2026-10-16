#!/usr/bin/env bash
# Plan task P1.4: the gym-pusht benchmark (6 trainings x 3 seeds, 100 k steps each, then every
# inference variant on the same 50 episodes). Idempotent: re-run to resume after an interruption.
#
# Usage:  nohup scripts/benchmark_gym.sh > outputs/benchmark_gym.log 2>&1 &
#         scripts/benchmark_gym.sh --only flow_matching --seeds 1      # a subset
#         movensys-benchmark run --list                                # the matrix
# Afterwards:  movensys-compare --write-plan     # results/gym/R1.md, the plot, plan Table R1
set -eo pipefail   # no -u: /opt/ros/jazzy/setup.bash references unset variables
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source env.bash
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p outputs
movensys-benchmark run "$@"
