#!/usr/bin/env bash
# Plan task P0.2 / Appendix A: uv-managed venv on system Python 3.12 with --system-site-packages
# so Jazzy's rclpy (built for 3.12) is importable next to LeRobot.
# Works on the x86 workstation (RTX 5090, cu128) and on a Jetson Thor (aarch64, JetPack 7, cu130).
set -eo pipefail   # no -u: /opt/ros/jazzy/setup.bash references unset variables
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROS_SETUP=/opt/ros/jazzy/setup.bash
if [ ! -f "$ROS_SETUP" ]; then
    cat >&2 <<MSG
!! ROS 2 Jazzy not found ($ROS_SETUP). Install it first (needs sudo):

   sudo apt update && sudo apt install -y software-properties-common curl
   sudo add-apt-repository -y universe
   ROS_APT_VER=\$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F'"' '{print \$4}')
   curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/\${ROS_APT_VER}/ros2-apt-source_\${ROS_APT_VER}.\$(. /etc/os-release && echo \$VERSION_CODENAME)_all.deb"
   sudo dpkg -i /tmp/ros2-apt-source.deb
   sudo apt update && sudo apt install -y ros-jazzy-ros-base python3-colcon-common-extensions

Then re-run $0.
MSG
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo ">> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Pick the torch wheel index for this machine.
if [ -f /etc/nv_tegra_release ] && [ "$(uname -m)" = "aarch64" ]; then
    TORCH_INDEX=https://download.pytorch.org/whl/cu130   # Jetson Thor: sm_110 needs CUDA 13
    echo ">> Jetson detected ($(head -1 /etc/nv_tegra_release | cut -d, -f1-2 | sed 's/^# //')): torch from cu130"
else
    TORCH_INDEX=https://download.pytorch.org/whl/cu128   # RTX 5090: sm_120 needs cu128
    echo ">> x86/dGPU host: torch from cu128"
fi

# shellcheck disable=SC1090
source "$ROS_SETUP"

[ -d .venv ] || uv venv --python 3.12 --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> torch ($TORCH_INDEX)"
# Bounds match LeRobot 0.6.1 (torch<2.12, torchvision<0.27) so the editable install keeps this build.
uv pip install "torch>=2.7,<2.12" "torchvision>=0.22,<0.27" --index-url "$TORCH_INDEX"
echo ">> lerobot and the rest"
uv pip install -e ".[dev,viz]"

python - <<'PY'
import numpy, torch, lerobot, rclpy  # noqa: F401
print("cuda:", torch.cuda.is_available(), "| torch", torch.__version__,
      "| lerobot", lerobot.__version__, "| numpy", numpy.__version__, "| rclpy OK")
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
PY
