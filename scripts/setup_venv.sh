#!/usr/bin/env bash
# Plan task P0.2 / Appendix A: uv-managed venv on system Python 3.12 with --system-site-packages
# so Jazzy's rclpy (built for 3.12) is importable next to LeRobot.
set -eo pipefail   # no -u: /opt/ros/jazzy/setup.bash references unset variables
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo ">> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

[ -d .venv ] || uv venv --python 3.12 --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> torch (cu128: the RTX 5090 is sm_120 and needs the cu128 wheels)"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo ">> lerobot and the rest"
uv pip install -e ".[dev,viz]"

python - <<'PY'
import numpy, torch, lerobot, rclpy  # noqa: F401
print("cuda:", torch.cuda.is_available(), "| torch", torch.__version__,
      "| lerobot", lerobot.__version__, "| numpy", numpy.__version__, "| rclpy OK")
PY
