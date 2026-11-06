Draccus YAML fragments for `movensys-train` / `movensys-eval`.

| File | Content |
|---|---|
| `pusht_gym.yaml` | stage 1 task: dataset `lerobot/pusht`, env `pusht`, 100 k steps, batch 64, 50-episode eval |
| `diffusion.yaml` | Diffusion Policy with the Phase 1 fairness settings |
| `flow_matching.yaml` | Flow Matching (`flow_matching_unet`) with the same settings |
| `act.yaml` | ACT default variant; matched / no-VAE variants are CLI overrides |
| `pusht_sim.yaml` | stage 2 task: dataset `movensys/pusht_cr3a_sim_v3` (episodes 60 to 64 held out), 256 px frames resized to 96 (crop 84), 20 k steps, no env eval |
| `pusht_real.yaml` | added in Phase 4 |

`--config_path` accepts several files separated by commas; later files deep-merge over
earlier ones and CLI flags override both, e.g.

```
movensys-train --config_path=movensys_imitation/configs/pusht_gym.yaml,movensys_imitation/configs/flow_matching.yaml \
    --seed=1 --output_dir=outputs/gym/flow_matching_s1
```

Hyperparameters are frozen after plan task P1.5; later phases change only `dataset` and `env`.
