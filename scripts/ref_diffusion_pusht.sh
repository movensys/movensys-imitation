#!/usr/bin/env bash
# Plan task P1.3: reference run. Diffusion Policy on lerobot/pusht for 200 k steps with the
# lerobot/diffusion_pusht model-card configuration, then 50 gym-pusht eval episodes.
# Done when the final success rate is within the model card's band (65.4 % on 500 episodes).
#
# LeRobot 0.6.1 changed the Diffusion defaults (horizon 64/32, no crop, ImageNet-pretrained
# BatchNorm encoder, one encoder per camera). The --policy.* flags below restore the model-card
# values, which are also the Phase 1 fairness settings in plan Section 6 (84 x 84 crop, 16 / 8).
#
# Usage:  scripts/ref_diffusion_pusht.sh [extra lerobot flags]
#   nohup scripts/ref_diffusion_pusht.sh > outputs/ref_diffusion_pusht.log 2>&1 &
#   RESUME=1 scripts/ref_diffusion_pusht.sh      # continue from the last checkpoint
set -eo pipefail   # no -u: /opt/ros/jazzy/setup.bash references unset variables
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source env.bash
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export MUJOCO_GL=${MUJOCO_GL:-egl}

OUT=${OUT:-outputs/ref_diffusion_pusht}
STEPS=${STEPS:-200000}
SEED=${SEED:-100000}          # the model card's seed
N_EVAL=${N_EVAL:-50}

if [ "${RESUME:-0}" = "1" ]; then
    movensys-train --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
        --resume=true "$@"
else
    mkdir -p "$(dirname "$OUT")"
    movensys-train --policy.type=diffusion --env.type=pusht --dataset.repo_id=lerobot/pusht \
      --seed="$SEED" --steps="$STEPS" --batch_size=64 \
      --policy.horizon=16 --policy.n_action_steps=8 --policy.drop_n_last_frames=7 \
      --policy.crop_shape="[84,84]" \
      --policy.pretrained_backbone_weights=null --policy.use_group_norm=true \
      --policy.use_separate_rgb_encoder_per_camera=false \
      --env_eval_freq=25000 --save_freq=25000 --log_freq=200 \
      --eval.n_episodes="$N_EVAL" --eval.batch_size="$N_EVAL" --eval.use_async_envs=false \
      --policy.push_to_hub=false --wandb.enable=false --output_dir="$OUT" "$@"
fi

movensys-eval --policy.path="$OUT/checkpoints/last/pretrained_model" \
  --env.type=pusht --eval.n_episodes="$N_EVAL" --eval.batch_size="$N_EVAL" \
  --eval.use_async_envs=false --seed="$SEED" --output_dir="$OUT/eval_final"
