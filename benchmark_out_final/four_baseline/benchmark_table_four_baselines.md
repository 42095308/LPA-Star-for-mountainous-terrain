# Four-Baseline Comparison Table

- Trials: `50`, seed: `20260309`, blocked edges: `4`

| Method | Planning Time (ms) | Path Length (km) | Path Cost | Energy (kJ) | Success Rate |
|---|---:|---:|---:|---:|---:|
| B1 Voxel | 384.06 | 7.703 | 63.8571 | 309.43 | 98.0% |
| B2 GlobalA* | 19.01 | 6.412 | 10.7701 | 268.87 | 100.0% |
| B3 FlatLPA* | 2.60 | 6.445 | 12.1773 | 232.79 | 100.0% |
| B4 Proposed | 47.52 | 6.412 | 10.7701 | 268.87 | 100.0% |

Note: All four baselines use the same multi-objective weighted cost (α·Time + β·Energy + γ·Risk). B1 cost is computed post-hoc along its Dijkstra path. Cross-baseline cost comparison is valid for the same graph scale.
