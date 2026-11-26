# 1. Launching the policy
```
source env.bash
ros2 launch movensys_imitation_ros run_policy.launch.py mode:=real \
    pretrained:=outputs/real/diffusion_v1_231ep_s1/checkpoints/100000/pretrained_model \
    overrides:="--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10" \
    run_mode:=slow dataset:=eval_diffusion_v1_231ep_s1 seed_start:=2000 n_episodes:=15 \
    auto_start:=true

```


# 2. the viewer
```
python -m movensys_imitation_ros.score_viewer
```

```
python -m movensys_imitation_ros.collect_keys      # s starts the first episode; a aborts
```

ros2 topic echo /policy_runner/state               # idle | running:<n> | stale | fault:<why>