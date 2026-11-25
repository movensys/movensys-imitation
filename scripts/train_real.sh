#!/usr/bin/env bash
# Train the policies on the real CR3A dataset (plan P4.6), one after the other.
#   scripts/train_real.sh [seed] [policy ...]      default: seed 1, diffusion flow_matching act
# Output: outputs/real/<policy>_v1_s<seed>/ with train.log next to the checkpoints. A policy
# whose output directory already exists is skipped (LeRobot refuses to overwrite it).
# Runs at nice 10 so the robot stack on the same machine keeps its CPU.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export HF_HUB_OFFLINE=1

seed="${1:-1}"
shift || true
policies=("$@")
[ ${#policies[@]} -eq 0 ] && policies=(diffusion flow_matching act)

mkdir -p outputs/real
for policy in "${policies[@]}"; do
    out="outputs/real/${policy}_v1_s${seed}"
    if [ -e "$out" ]; then
        echo "$(date '+%F %T') skip $policy: $out exists"
        continue
    fi
    task=movensys_imitation/configs/pusht_real.yaml
    if [ "$policy" = act ]; then
        # ACTConfig has no resize_shape / crop_ratio: ACT trains on the recorder's 256 x 256
        # frame as it is, so the task fragment is used without its policy block.
        task="outputs/real/pusht_real_nopolicy.yaml"
        python -c "import sys, yaml; d = yaml.safe_load(open(sys.argv[1])); d.pop('policy', None); yaml.safe_dump(d, open(sys.argv[2], 'w'))" \
            movensys_imitation/configs/pusht_real.yaml "$task"
    fi
    echo "$(date '+%F %T') start $policy seed $seed -> $out"
    nice -n 10 movensys-train \
        --config_path="movensys_imitation/configs/${policy}.yaml,${task}" \
        --seed="$seed" --output_dir="$out" > "outputs/real/${policy}_v1_s${seed}.log" 2>&1
    status=$?
    mv "outputs/real/${policy}_v1_s${seed}.log" "$out/train.log" 2>/dev/null
    echo "$(date '+%F %T') done $policy seed $seed, exit $status"
done
