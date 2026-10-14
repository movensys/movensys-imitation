# movensys-imitation

ACT, Diffusion Policy and Flow Matching compared on PushT with the Dobot CR3A. Demonstrations are
collected with Meta Quest teleoperation through MoveIt Servo; commands are executed through
WMX-R2, first in Isaac Sim and then on the real arm.

The plan, milestones and result tables are in [`plan/plan.md`](plan/plan.md).

## Layout

| Path | Content |
|---|---|
| `movensys_imitation/` | Python library: policies (`flow_matching_unet`), PushT geometry and scoring, dataset tools, train/eval entry points |
| `movensys_imitation_ros/` | ament_python ROS 2 package: planar guard, episode recorder and manager, scorer, policy runner |
| `sim/isaacsim/` | Isaac Sim PushT scenes and assets |
| `hardware/` | printable T block, pusher tip, target sheet |
| `results/` | committed evaluation CSVs |
| `doc/` | setup and results documents |

## Setup (plan task P0.2)

```bash
cd ~/workspaces/movensys-imitation
scripts/setup_venv.sh          # uv venv on system Python 3.12, torch cu128, LeRobot 0.6.1
source env.bash                # ROS_DOMAIN_ID=70, Jazzy, movensys_ws, .venv
scripts/smoke_test_gym.sh      # 500 Diffusion steps on lerobot/pusht + 5 gym-pusht eval episodes
```

Data and checkpoints go to `datasets/` and `outputs/`, which are not committed.
