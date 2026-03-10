# Benchmark Table

- Trials requested: `50`, random seed: `20260309`, blocked edges per trial: `2`
- B1 voxel config: `xy_step=125m`, `agl_step=5m`, timeout `8.0s`

## Per-baseline summary

| Baseline | Success | Replan ms (mean+/-std) | P50/P95 ms | Expanded (mean) | Cost (mean) | Energy kJ (mean) | Length km (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered | 50/50 (100.0%) | 59.51+/-79.10 | 28.64/202.94 | 640.2 | 13.2239 | 243.94 | 6.344 |
| B2_GlobalAstar_Layered | 50/50 (100.0%) | 28.69+/-18.87 | 26.47/59.23 | 2218.7 | 13.2239 | 243.94 | 6.344 |
| B3_LPA_SingleLayer | 50/50 (100.0%) | 1.46+/-3.53 | 0.27/7.41 | 7.6 | 16.1184 | 222.70 | 6.436 |
| B1_Voxel_Dijkstra | 50/50 (100.0%) | 492.90+/-267.86 | 496.58/901.21 | 59459.6 | 7.3023 | nan | 7.302 |

## Paired significance checks

| Pair | Metric | N paired | Mean A | Mean B | Median(B/A) | p-value |
|---|---|---:|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered vs B2_GlobalAstar_Layered | replan_ms | 50 | 59.5142 | 28.6883 | 0.610 | 2.682e-03 |
| B4_Proposed_LPA_Layered vs B3_LPA_SingleLayer | path_cost | 50 | 13.2239 | 16.1184 | 1.220 | 1.020e-20 |
| B4_Proposed_LPA_Layered vs B1_Voxel_Dijkstra | replan_ms | 50 | 59.5142 | 492.8986 | 12.129 | 4.849e-17 |
