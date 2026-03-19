# Four-Baseline Comparison Table

- Trials: `50`, seed: `20260309`, blocked edges: `4`

| Method | Planning Time (ms) | Path Length (km) | Path Cost | Success Rate |
|---|---:|---:|---:|---:|
| B1 Voxel | 532.53 | 7.703 | 7.7025 | 98.0% |
| B2 GlobalA* | 30.14 | 6.412 | 10.7701 | 100.0% |
| B3 FlatLPA* | 3.90 | 6.445 | 12.1773 | 100.0% |
| B4 Proposed | 69.30 | 6.412 | 10.7701 | 100.0% |

Note: B1 path cost follows its voxel baseline output (approximately same scale as length), while B2/B3/B4 path cost is the weighted multi-objective cost.
