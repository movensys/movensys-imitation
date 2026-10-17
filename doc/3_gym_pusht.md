# 3. Stage 1: gym-pusht

Plan Phase 1: the Flow Matching policy, the three-policy benchmark on `lerobot/pusht`, and the
conversion of that dataset into the recorder schema. Everything runs in the host venv
(`source env.bash`); `SDL_VIDEODRIVER=dummy` and `MUJOCO_GL=egl` are set by the scripts.

## Flow Matching policy (P1.1)

`movensys_imitation/policies/flow_matching_unet/` registers `flow_matching_unet` with LeRobot.
`FlowMatchingUnetConfig` subclasses `DiffusionConfig`; `FlowMatchingUnetPolicy` subclasses
`DiffusionPolicy` and swaps only the model: `DiffusionRgbEncoder` and `DiffusionConditionalUnet1d`
are imported from LeRobot, the loss is the OT flow-matching objective and the sampler is a fixed-step
Euler (or midpoint) integrator. Config additions: `fm_num_steps`, `fm_solver`, `fm_time_sampling`
(`uniform` or pi0's `beta`), `fm_time_embed_scale`.

Tests (`pytest movensys_imitation/tests/test_flow_matching_unet.py`): registration, shapes, loss
decreases on synthetic data, 1- and 16-step samplers finite and seed-deterministic, OT path and
Euler / midpoint integrator equal to Meta's `flow_matching` library on a 2-D toy field.

## Entry points and configs (P1.2)

`movensys-train` / `movensys-eval` wrap LeRobot's scripts with the project policies registered.
`--config_path` takes a comma-separated list of YAML fragments (later files override earlier ones):

```
movensys-train --config_path=movensys_imitation/configs/pusht_gym.yaml,movensys_imitation/configs/flow_matching.yaml \
    --seed=1 --output_dir=outputs/gym/flow_matching_s1
movensys-eval --policy.path=outputs/gym/flow_matching_s1/checkpoints/last/pretrained_model \
    --policy.fm_num_steps=4 --env.type=pusht --eval.n_episodes=50 --eval.batch_size=50 \
    --eval.use_async_envs=false --seed=100000 --output_dir=outputs/gym/flow_matching_s1/eval_euler4
```

`movensys_imitation/configs/`: `pusht_gym.yaml` (task and budget), `diffusion.yaml`,
`flow_matching.yaml`, `act.yaml` (policies, Phase 1 fairness settings from plan Section 6).

## Reference run (P1.3)

`scripts/ref_diffusion_pusht.sh`: Diffusion, model-card configuration, 200 k steps, seed 100000.

| Eval (50 episodes, seeds 100000+) | Success 0.95 | Success 0.90 | Avg max coverage |
|---|---|---|---|
| final standalone eval | 58 % | 78 % | 0.87 |
| in-training eval at 200 k | 64 % | | |
| model card (500 episodes) | 65.4 % | | |

In-training success by checkpoint (25 k to 200 k): 34, 50, 58, 60, 62, 60, 56, 64 %. The policy
plateaus near 100 k steps, which supports the 100 k budget of the benchmark. 4.5 h on the RTX 5090
at about 12 steps/s, peak 5.0 GB allocated.

Reading LeRobot's numbers: gym-pusht's reward is `clip(coverage / 0.95, 0, 1)`, so
`avg_max_reward` x 0.95 is the average max coverage and `pc_success` is the 0.95-coverage
success rate. The 0.90 column is computed from `eval_info.json` by `movensys-benchmark`.

## Benchmark (P1.4)

```
nohup scripts/benchmark_gym.sh > outputs/benchmark_gym.log 2>&1 &     # whole matrix, resumable
movensys-benchmark run --list                                          # the matrix
```

Matrix (`movensys_imitation/benchmark.py`): 100 k steps, batch 64, seeds 1, 2, 3; every
checkpoint evaluated on the same 50 episodes (seeds 100000 to 100049).

| Training | Inference variants |
|---|---|
| `diffusion` | DDIM 10, DDPM 100 |
| `flow_matching` | Euler 1, 2, 4, 8, 16 |
| `act_default` (chunk 100, temporal ensembling), `act_default_novae` | default |
| `act_matched` (16 / 8, no ensembling), `act_matched_novae` | default |

Per run: `outputs/gym/<training>_s<seed>/` (checkpoints, `eval_<variant>/`), the train log next to
it, and `results/gym/<training>_<variant>_s<seed>.csv` (one row per episode: sum reward, max reward,
max coverage, success at 0.95 and 0.90) with a JSON sidecar (train seconds, peak VRAM from the
train log, latency, parameter count). Latency is one `select_action` call after `reset()` with a
batch of one on the GPU, pre/post-processing excluded, mean and p95 over 100 calls. ACT learning
rate: LeRobot default 1e-5 at batch 64 (plan Section 6 allows one adjustment; none made yet).

## Tables and plots (P1.5)

```
movensys-compare                # results/gym/R1.md and results/gym/r1_success_vs_steps.png
movensys-compare --write-plan   # also rewrites Table R1 in plan/plan.md
```

Mean and sample standard deviation over seeds. Hyperparameters freeze when the table is filled;
`movensys_imitation/configs/` is then the frozen set and later phases change only `dataset` and `env`.

## Dataset conversion (P1.6)

```
movensys-convert-gym                   # datasets/movensys/pusht_gym (206 episodes, 256 x 256 video)
movensys-train --config_path=movensys_imitation/configs/diffusion.yaml \
    --dataset.repo_id=movensys/pusht_gym --dataset.root=datasets/movensys/pusht_gym \
    --policy.resize_shape="[96,96]" --policy.crop_ratio=0.875 --steps=1000 --env_eval_freq=0 \
    --output_dir=outputs/converted_gym_diffusion
```

`observation.images.top` is the 96 x 96 gym frame upscaled to 256 x 256; state and action stay in
pixel units; `observation.joints` and `block_pose` are zeros (not in `lerobot/pusht`); `coverage` is
`next.reward` x 0.95. With `resize_shape=[96,96]` and `crop_ratio=0.875` the policy sees the same
84 x 84 crop as on the original data. No env eval on this set: the gym env's image key differs and
LeRobot only renames features for pretrained checkpoints.

## Known issues

* `torchcodec` cannot load its FFmpeg libraries on this host and LeRobot falls back to `pyav`; the
  long warning at every start is harmless.
* `PreTrainedConfig.from_pretrained(..., cli_overrides=[...])` takes policy-level flags
  (`--fm_num_steps=4`, not `--policy.fm_num_steps=4`) and ignores unknown keywords silently;
  `benchmark.measure_latency` checks each override was applied.

## Gripper control in isaac scene
The gripper is two prismatic finger joints in the robot articulation, picker_1_joint and picker_2_joint, each with a position drive from 0 (open) to 0.045 m (closed). I verified the direction headless in the current scene:

Commanded position	Finger distance
0.000	100 mm (open)
0.045	10 mm (fully closed)
Three ways to close it:

1. Over ROS 2, while the scene plays. The joint-state graph feeds /joint_command into the articulation controller, and it accepts a partial joint list, so you can command only the fingers:


source /opt/ros/jazzy/setup.bash
ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
  "{name: [picker_1_joint, picker_2_joint], position: [0.040, 0.040]}"
Use 0.040 rather than 0.045 to stop at a 20 mm gap, which is the tip diameter. Open again with 0.0. The positions are echoed back on /joint_states.