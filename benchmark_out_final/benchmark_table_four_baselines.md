# Four-Baseline Comparison Table

- Trials: `200`, seed: `20260313`, blocked edges: `4`

| Method | Planning Time (ms) | Path Length (km) | Path Cost | Energy (kJ) | Success Rate |
|---|---:|---:|---:|---:|---:|
| B1 Voxel | 362.11 | 7.218 | 60.3283 | 289.92 | 99.0% |
| B2 GlobalA* | 17.84 | 6.055 | 10.1275 | 253.58 | 100.0% |
| B3 FlatLPA* | 5.29 | 6.080 | 11.4335 | 219.69 | 100.0% |
| B4 Proposed | 44.10 | 6.055 | 10.1275 | 253.58 | 100.0% |

Note: All four baselines use the same multi-objective weighted cost (α·Time + β·Energy + γ·Risk). B1 cost is computed post-hoc along its Dijkstra path. Cross-baseline cost comparison is valid for the same graph scale.
