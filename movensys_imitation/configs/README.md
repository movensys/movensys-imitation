Draccus YAML fragments for `movensys-train` / `movensys-eval`.

| File | Content |
|---|---|
| `pusht_gym.yaml` | stage 1 task: dataset `lerobot/pusht`, env `pusht`, 100 k steps, batch 64, 50-episode eval |
| `diffusion.yaml` | Diffusion Policy with the Phase 1 fairness settings |
| `flow_matching.yaml` | Flow Matching (`flow_matching_unet`) with the same settings |
| `act.yaml` | ACT default variant; matched / no-VAE variants are CLI overrides |
| `pusht_sim.yaml`, `pusht_real.yaml` | added in Phase 2 / 4 |

`--config_path` accepts several files separated by commas; later files deep-merge over
earlier ones and CLI flags override both, e.g.

```
movensys-train --config_path=movensys_imitation/configs/pusht_gym.yaml,movensys_imitation/configs/flow_matching.yaml \
    --seed=1 --output_dir=outputs/gym/flow_matching_s1
```

Hyperparameters are frozen after plan task P1.5; later phases change only `dataset` and `env`.
