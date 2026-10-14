#!/usr/bin/env bash
# Plan task P0.2: 500 steps of Diffusion Policy on lerobot/pusht, then 5 gym-pusht eval episodes.
# Proves cu128 torch on the 5090, LeRobot 0.6.1, gym-pusht and rclpy co-exist in one venv.
set -eo pipefail   # no -u: /opt/ros/jazzy/setup.bash references unset variables
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source env.bash
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}   # gym-pusht renders with pygame; no display needed
export MUJOCO_GL=${MUJOCO_GL:-egl}
OUT=outputs/smoke_diffusion
rm -rf "$OUT"
movensys-train --policy.type=diffusion --env.type=pusht --dataset.repo_id=lerobot/pusht \
  --steps=500 --batch_size=64 --env_eval_freq=0 --save_freq=500 --log_freq=50 \
  --policy.push_to_hub=false --wandb.enable=false --output_dir="$OUT" "$@"
movensys-eval --policy.path="$OUT/checkpoints/last/pretrained_model" \
  --env.type=pusht --eval.n_episodes=5 --eval.batch_size=5 --eval.use_async_envs=false \
  --output_dir="$OUT/eval"   # sync envs: see doc/1_setup.md, known issues
