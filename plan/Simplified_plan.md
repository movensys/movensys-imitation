# movensys-imitation — Project Roadmap

**Repository:** `movensys/movensys-imitation`

## 1. Project goal

Compare three imitation-learning policies on the same PushT task using the Dobot CR3A:

- **ACT** — Action Chunking with Transformers
- **Diffusion Policy** — Diffusion-based action generation
- **Flow Matching** — Our implementation using LeRobot's diffusion encoder and UNet

Demonstrations will be collected using Meta Quest teleoperation through MoveIt Servo. Commands will be executed through WMX-R2, first in Isaac Sim and then on the real robot.

### Final deliverables

1. One repository containing training, data collection, evaluation, and deployment tools.
2. Three policies trained on comparable datasets and budgets.
3. A working simulation-to-hardware pipeline.
4. Quantitative results and videos comparing all three policies.
5. Setup instructions that another engineer can follow.

## 2. Project workflow

```text
Stage 1: Gym-PushT
    Validate training and evaluation
             ↓
Stage 2: Isaac Sim
    Build the CR3A PushT task
    Collect demonstrations and evaluate policies
             ↓
Stage 3: Hardware-in-the-loop
    Validate the WMX-R2 streaming path
             ↓
Stage 4: Real CR3A
    Collect real demonstrations
    Train and evaluate all three policies
             ↓
Stage 5: Results
    Compare policies and document findings
```

**Important:** Do not begin autonomous hardware experiments until the streaming controller, workspace limits, emergency stop, and low-speed motion have been verified.

## 3. Master task tracker

Update this table during every project review. Detailed checklists for each phase follow below.

| ID | Milestone | Target | Status |
|---|---|---|---|
| M0 | Environment, teleoperation, and WMX streaming verified | Week 1 | Not started |
| M1 | Three policies benchmarked on Gym-PushT | Week 3 | Not started |
| M2 | Simulation dataset and policy evaluation completed | Week 6 | Not started |
| M3 | Hardware-in-the-loop test passed | Week 6 | Not started |
| M4 | Real-robot evaluation completed | Week 9 | Not started |
| M5 | Final comparison and documentation completed | Week 10 | Not started |

Status values: `Not started`, `In progress`, `Blocked`, `Done`.

## 4. Phase 0 — Environment and prerequisites

**Target:** Week 1

Goal: Prove that the existing robotics stack, ML environment, and WMX streaming path work before developing new functionality.

### Tasks

- [ ] P0.1 Create the repository with the basic Python package, ROS 2 package, documentation, and tests.
- [ ] P0.2 Set up the Python 3.12 virtual environment and install compatible PyTorch CUDA and LeRobot dependencies.
- [ ] P0.3 Run a short Diffusion Policy training and evaluation test on Gym-PushT.
- [ ] P0.4 Switch `movensys-manipulator` to `devel` and build the required ROS 2 containers.
- [ ] P0.5 Verify Quest pose publishing and teleoperation in Isaac Sim.
- [ ] P0.6 Select a WMX-R2 streaming controller and verify six-axis streaming on the WMX simulation platform.
- [ ] P0.7 Fix the gripper controller's hard-coded SDK path.

### Exit criteria

- Gym-PushT training and evaluation run successfully.
- Quest teleoperation moves the CR3A in Isaac Sim.
- WMX simulation axes follow streamed joint commands.
- The selected streaming controller and its remaining limitations are documented.

**Main blocker:** WMX-R2 streaming support. Resolve this early because it affects the hardware stages.

## 5. Phase 1 — Policy implementation and benchmark

**Target:** Weeks 2–3

Goal: Establish a reliable benchmark and implement Flow Matching before introducing robot-specific complexity.

### Tasks

- [ ] P1.1 Train and evaluate the standard LeRobot Diffusion Policy on Gym-PushT.
- [ ] P1.2 Implement `flow_matching_unet` using LeRobot's diffusion encoder and conditional UNet.
- [ ] P1.3 Add tests for output shapes, loss behavior, and deterministic sampling.
- [ ] P1.4 Add unified training and evaluation commands for ACT, Diffusion Policy, and Flow Matching.
- [ ] P1.5 Benchmark all three policies using fixed evaluation seeds.
- [ ] P1.6 Record success rate, training time, inference latency, and GPU memory usage.
- [ ] P1.7 Freeze the initial training configurations.

### Benchmark rules

Use the same dataset, evaluation seeds, and comparable training budgets.

- Three training seeds per policy.
- Initial training budget: 100,000 gradient steps per run.
- Batch size: 64, where supported.
- Fifty evaluation episodes per trained checkpoint.
- Record mean and standard deviation across training seeds.

Allow policy-specific settings when necessary, but document them rather than claiming perfectly identical configurations.

### Exit criteria

- All three policies train and evaluate through the same workflow.
- Diffusion Policy produces results in a reasonable range relative to the published Gym-PushT benchmark.
- Flow Matching passes its unit tests and produces valid actions.
- The initial configurations are committed and ready for the robot experiments.

## 6. Phase 2 — PushT in Isaac Sim

**Target:** Weeks 3–6

Goal: Validate the complete data collection and inference pipeline on the simulated CR3A.

### Tasks

- [ ] P2.1 Create the PushT scene using the existing CR3A simulation scene.
- [ ] P2.2 Add the T block, target outline, pusher tip, overhead camera, and block-pose interface.
- [ ] P2.3 Implement `planar_guard` to constrain the pusher to the permitted workspace and height.
- [ ] P2.4 Implement the episode recorder and episode manager.
- [ ] P2.5 Collect 10 pilot episodes and verify the dataset.
- [ ] P2.6 Train ACT, Diffusion Policy, and Flow Matching on the simulation dataset.
- [ ] P2.7 Implement policy execution with shadow, slow, and normal modes.
- [ ] P2.8 Collect approximately 150–200 simulation demonstrations.
- [ ] P2.9 Evaluate the policies on the same 50 start states and save the results.

### Initial task configuration

| Parameter | Value |
|---|---|
| Task | Planar PushT |
| Robot | Dobot CR3A |
| Observation | Overhead RGB image and pusher XY position |
| Action | Absolute pusher XY target |
| Policy frequency | 10 Hz |
| Guard / teleoperation | 50 Hz target updates |
| Episode limit | 300 policy steps |
| Primary success threshold | Coverage ≥ 0.95 |

### Exit criteria

- The robot completes the PushT task in Isaac Sim through MoveIt Servo.
- Demonstrations can be recorded, saved, and replayed.
- All three policies execute through the same guarded control path.
- Simulation results are available for the fixed evaluation set.

## 7. Phase 3 — Hardware-in-the-loop

**Target:** Week 6

Goal: Validate command streaming through WMX-R2 before moving the experiment to physical hardware.

### Tasks

- [ ] P3.1 Create the HIL scene with the required ROS namespace and joint interfaces.
- [ ] P3.2 Connect MoveIt Servo to the WMX simulation platform.
- [ ] P3.3 Relay WMX-reported joint states to Isaac Sim.
- [ ] P3.4 Run 10 teleoperation episodes and 10 policy episodes.
- [ ] P3.5 Measure joint tracking error, command timing, and end-to-end latency.
- [ ] P3.6 Resolve streaming and timing problems before enabling real hardware.

### Exit criteria

- A complete policy episode runs successfully through the WMX-R2 streaming path.
- Joint commands and reported states behave as expected.
- The streaming controller is reviewed and merged into the agreed branch.

## 8. Phase 4 — Real CR3A

**Target:** Weeks 7–9

Goal: Collect physical demonstrations, train the policies, and measure real-world performance.

### Tasks

- [ ] P4.1 Prepare the printed T block, pusher tip, target sheet, and overhead camera mount.
- [ ] P4.2 Verify EtherCAT, the WMX3 license, the emergency stop, and joint soft limits.
- [ ] P4.3 Configure the overhead camera and AprilTag-based block-pose measurement.
- [ ] P4.4 Calibrate the workspace and verify the pusher height and motion limits.
- [ ] P4.5 Test teleoperation at reduced speed and verify every safety layer.
- [ ] P4.6 Collect 100–150 real demonstrations under at least two lighting conditions.
- [ ] P4.7 Train all three policies on the real dataset, using three seeds per policy.
- [ ] P4.8 Evaluate the selected checkpoint for each policy on the same 20 start states.
- [ ] P4.9 Record videos, success outcomes, motion quality, and controller tracking data.

### Hardware safety requirements

Before every autonomous run:

- Confirm the physical emergency stop is accessible and working.
- Enable and verify WMX joint soft limits.
- Verify the workspace clamp, fixed-height constraint, command watchdog, and motion limits.
- Confirm that releasing the dead-man control stops autonomous command updates safely.
- Run the checkpoint in shadow mode, then reduced-speed mode, before normal operation.
- Keep an operator ready to stop the robot throughout the experiment.

Never use the CR3A homing service to redefine its absolute-encoder zero.

### Exit criteria

- All three policies have been evaluated on the physical robot.
- Each policy has 20 recorded trials on the predefined start-state set.
- Success rate and motion-quality metrics are calculated consistently.
- Safety checks and experiment procedures are documented.

## 9. Phase 5 — Results and final report

**Target:** Week 10

### Tasks

- [ ] P5.1 Generate comparison tables and plots from evaluation CSV files.
- [ ] P5.2 Compare simulation and real-robot performance.
- [ ] P5.3 Categorize failures, such as overshoot, oscillation, incorrect pushing direction, and getting stuck.
- [ ] P5.4 Summarize training cost, inference latency, motion quality, and success rate.
- [ ] P5.5 Prepare a short demonstration video and management summary.
- [ ] P5.6 Finalize setup instructions and repository documentation.

### Final metrics

| Metric | Purpose |
|---|---|
| Success rate | Percentage of episodes reaching the target |
| Maximum coverage | Measures how closely the T reaches the target |
| Time to success | Measures task completion speed |
| Path length | Measures pusher travel |
| Motion smoothness | Measures changes in commanded motion |
| Joint tracking error | Measures controller tracking on hardware |
| Inference latency | Measures policy computation cost |
| Training time and VRAM | Measures training cost |

Report mean and standard deviation across training seeds where applicable. Report hardware trials separately from training-seed variability.

## 10. Timeline

| Week | Main focus | Milestone |
|---|---|---|
| 1 | Environment, Quest, Servo, WMX streaming | M0 |
| 2 | Flow Matching implementation | — |
| 3 | Gym-PushT benchmark and frozen configurations | M1 |
| 4 | Isaac Sim scene, guard, recorder, pilot data | — |
| 5 | Simulation data collection and training | — |
| 6 | Simulation evaluation and HIL | M2, M3 |
| 7 | Hardware preparation and low-speed testing | — |
| 8 | Real data collection and training | — |
| 9 | Real-robot evaluation | M4 |
| 10 | Analysis, videos, and documentation | M5 |

These are target dates, not guaranteed completion dates. A blocked prerequisite takes priority over starting the next phase.

## 11. Optional work — Only after the main experiment

Do not allow optional experiments to delay the three-policy comparison.

Priority order:

1. Test Flow Matching inference with 1, 2, 4, 8, and 16 solver steps.
2. Compare different action horizons.
3. Add LeRobot real-time chunking.
4. Add the DiT Flow Matching baseline.
5. Fine-tune SmolVLA.
6. Test additional camera views or other tasks.

## 12. Key technical decisions

| Area | Decision |
|---|---|
| Robot | Dobot CR3A |
| Robot execution | WMX-R2 |
| Motion interface | MoveIt Servo with a shared planar guard |
| Teleoperation | Meta Quest through the existing ROS 2 teleoperation stack |
| Simulator | Isaac Sim 5.1 |
| ML framework | LeRobot 0.6.1, subject to dependency verification |
| Python environment | Python 3.12 virtual environment |
| Dataset | LeRobot format |
| Primary task | PushT |
| Evaluation | Fixed start states, consistent scoring, saved CSVs and videos |
| Version control | One main project repository plus documented dependencies on sibling repositories |

## 13. Project review

At each review, answer only these five questions:

1. What tasks were completed since the previous review?
2. What is the next task?
3. What is blocking progress?
4. What evidence demonstrates that the completed task works?
5. Is the next milestone still achievable?

Keep detailed ROS topics, shell commands, controller branch notes, simulation instructions, and safety procedures in separate setup documents. The roadmap should explain **what to do, in what order, and how to know it is finished**.