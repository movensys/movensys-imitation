# 1. Setup

Host checked: Ubuntu 24.04, ROS 2 Jazzy (`/opt/ros/jazzy`), system Python 3.12.3, RTX 5090, Isaac Sim 5.1 in `~/isaac-sim`, WMX3 SDK in `/opt/wmx3`.

## 1.1 Python environment (plan task P0.2)

One uv-managed venv on the system Python 3.12 with `--system-site-packages`, so Jazzy's `rclpy` (built for 3.12) is importable next to LeRobot. The venv's own numpy 2.x shadows the system numpy 1.26; the apt `cv_bridge` is built against 1.26, so venv nodes decode images with numpy and never import `cv_bridge`.

```bash
cd ~/workspaces/movensys-imitation
scripts/setup_venv.sh        # installs uv if missing, creates .venv, torch cu128, lerobot 0.6.1, this package
source env.bash              # every terminal: ROS_DOMAIN_ID=0, Jazzy, movensys_ws overlay, .venv
scripts/smoke_test_gym.sh    # 500 Diffusion steps on lerobot/pusht, then 5 gym-pusht eval episodes
```

Notes.

- torch and torchvision come from the `cu128` index first; the RTX 5090 (sm_120) is not supported by the default PyPI CUDA build.
- `set -u` must not be used in scripts that source `/opt/ros/jazzy/setup.bash` (it references unset variables).
- The `ffmpeg` binary is not installed on the host and `sudo` needs a password. torchcodec only needs the shared libraries (`libavcodec.so.60` is present), so dataset reading works. Video encoding of recorded episodes (Phase 2) is done with PyAV, which ships its own ffmpeg.
- `SDL_VIDEODRIVER=dummy` lets gym-pusht render headless; `scripts/smoke_test_gym.sh` sets it.

Verified on 2026-10-13: `cuda: True | torch 2.11.0+cu128 | lerobot 0.6.1 | numpy 2.2.6 | rclpy OK`; the smoke test trains 500 Diffusion steps in 72 s (about 10 steps/s, 5 GB VRAM, loss 0.88 to 0.05) and evaluates 5 gym-pusht episodes in 17 s (`pc_success` 0.0 at 500 steps, as expected).

### Known issues found while setting up (all handled by the scripts and `pyproject.toml`)

| Issue | Cause | Fix in this repo |
|---|---|---|
| `numpy.dtype size changed` on `import lerobot` | apt `scipy` (built against numpy 1.26) leaks in through `--system-site-packages`; diffusers imports `scipy.stats` | `scipy` and `matplotlib` are venv dependencies so they shadow the apt builds. Rule: any compiled package a venv node needs must be installed in the venv. |
| `'accelerate' is required` at train time | LeRobot 0.6.1 moved the training loop behind the `training` extra | `lerobot[...,training]` in `pyproject.toml` |
| `pytest` fails with `PluginValidationError` from `launch_testing` | the apt ROS pytest plugins are incompatible with pytest >= 8 | `addopts = "-p no:launch_testing -p no:launch_ros"` in `pyproject.toml` |
| torchcodec cannot load (`libavdevice.so.60` missing) | the host has no `ffmpeg` package and no sudo | LeRobot falls back to PyAV automatically (warning only); videos still decode and encode |
| eval: `NamespaceNotFound: gym_pusht` in env workers | LeRobot builds `AsyncVectorEnv` with the forkserver context and only imports `gym_pusht` in the parent | `movensys-eval` preloads `gym_pusht` into the forkserver (`cli/_lerobot.py`) |
| eval: `'NoneType' object has no attribute 'send'` after a `task_description` warning | gymnasium 1.4 shuts async workers down when LeRobot probes a missing env attribute | pass `--eval.use_async_envs=false` (gym-pusht is cheap, sync is fine); re-test when LeRobot or gymnasium is bumped |

## 1.2 Domain ID and sim time

`ROS_DOMAIN_ID=0` is the project domain (decided 2026-10-24; `~/.bashrc` and `env.bash` both export 0, Isaac Sim's ROS 2 bridge uses the default 0 and `rmw_fastrtps_cpp`, which interoperates with the cyclonedds host and containers); both the manipulator and the teleop compose files pass `${ROS_DOMAIN_ID}` through from the host shell, so start containers from a shell that has it. Ignore the 73 in the manipulator README. `use_sim_time` is true in sim and HIL, false on the real robot.

## 1.3 Manipulator container (task P0.3)

_To be written by the controls engineer: `origin/devel` checkout, `ros-jazzy-moveit-servo` in the `isaac-ros_4.1` Dockerfile, build, `servo_node` check._

## 1.4 Isaac Sim path check (task P0.4)

_To be written: `3a_trajectory_simulation.usd`, `sim_bridge` + `moveit` with `use_sim_time:=true`, keyboard teleop._

## 1.5 Teleop container and Quest (task P0.5)

_To be written: CloudXR runtime from NGC, `quest_pose.launch.py`, WebXR client, planar teleop in sim. Known issues are in the teleop repo's `doc/1_setup.md` to `doc/3_quest_pose.md`._

## 1.6 wmx-r2 streaming controller and WMX simu platform (tasks P0.6, P0.7)

_To be written: controller comparison note, six-axis `simu_network.def`, launch line:_

```bash
sudo --preserve-env=PATH,AMENT_PREFIX_PATH,COLCON_PREFIX_PATH,PYTHONPATH,LD_LIBRARY_PATH,ROS_DISTRO,ROS_VERSION,ROS_PYTHON_VERSION,ROS_DOMAIN_ID,RMW_IMPLEMENTATION \
  bash -c "source /opt/ros/jazzy/setup.bash && source ~/workspaces/movensys_ws/install/setup.bash && \
           ros2 launch wmx_r2_package wmx_r2_cr3a_manipulator.launch.py"
```
