# Benchmark Table

- Trials requested: `50`, random seed: `20260309`, blocked edges per trial: `4`
- B1 voxel config: `xy_step=125m`, `agl_step=5m`, timeout `8.0s`

## Per-baseline summary

| Baseline | Success | Replan ms (mean+/-std) | P50/P95 ms | Expanded (mean) | Cost (mean) | Energy kJ (mean) | Length km (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered | 50/50 (100.0%) | 69.30+/-60.10 | 48.46/157.59 | 779.0 | 10.7701 | 268.87 | 6.412 |
| B2_GlobalAstar_Layered | 50/50 (100.0%) | 30.14+/-21.76 | 25.61/69.26 | 2124.3 | 10.7701 | 268.87 | 6.412 |
| B3_LPA_SingleLayer | 50/50 (100.0%) | 3.90+/-9.80 | 0.63/17.44 | 25.4 | 12.1773 | 232.79 | 6.445 |
| B1_Voxel_Dijkstra | 49/50 (98.0%) | 532.53+/-264.81 | 546.95/887.10 | 67447.7 | 7.7025 | nan | 7.703 |

## Paired significance checks

| Pair | Metric | N paired | Mean A | Mean B | Median(B/A) | p-value |
|---|---|---:|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered vs B2_GlobalAstar_Layered | replan_ms | 50 | 69.2998 | 30.1427 | 0.512 | 2.887e-07 |
| B4_Proposed_LPA_Layered vs B3_LPA_SingleLayer | path_cost | 50 | 10.7701 | 12.1773 | 1.131 | 5.505e-18 |
| B4_Proposed_LPA_Layered vs B1_Voxel_Dijkstra | replan_ms | 49 | 70.2231 | 532.5339 | 8.739 | 5.307e-19 |
