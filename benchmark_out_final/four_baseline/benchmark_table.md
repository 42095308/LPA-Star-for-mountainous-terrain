# Benchmark Table

- Trials requested: `50`, random seed: `20260309`, blocked edges per trial: `4`
- B1 voxel config: `xy_step=125m`, `agl_step=5m`, timeout `8.0s`

## Per-baseline summary

| Baseline | Success | Replan ms (mean+/-std) | P50/P95 ms | Expanded (mean) | Cost (mean) | Energy kJ (mean) | Length km (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered | 50/50 (100.0%) | 47.52+/-40.49 | 32.18/113.48 | 779.0 | 10.7701 | 268.87 | 6.412 |
| B2_GlobalAstar_Layered | 50/50 (100.0%) | 19.01+/-13.24 | 15.01/42.15 | 2124.3 | 10.7701 | 268.87 | 6.412 |
| B3_LPA_SingleLayer | 50/50 (100.0%) | 2.60+/-5.56 | 0.60/13.60 | 25.4 | 12.1773 | 232.79 | 6.445 |
| B1_Voxel_Dijkstra | 49/50 (98.0%) | 384.06+/-191.13 | 386.41/648.57 | 67447.7 | 63.8571 | 309.43 | 7.703 |

## Paired significance checks

| Pair | Metric | N paired | Mean A | Mean B | Median(B/A) | p-value |
|---|---|---:|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered vs B2_GlobalAstar_Layered | replan_ms | 50 | 47.5187 | 19.0123 | 0.429 | 6.508e-08 |
| B4_Proposed_LPA_Layered vs B3_LPA_SingleLayer | path_cost | 50 | 10.7701 | 12.1773 | 1.131 | 5.505e-18 |
| B4_Proposed_LPA_Layered vs B1_Voxel_Dijkstra | replan_ms | 49 | 48.1325 | 384.0596 | 9.123 | 7.203e-19 |
