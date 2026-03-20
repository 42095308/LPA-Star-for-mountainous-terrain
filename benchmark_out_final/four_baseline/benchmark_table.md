# Benchmark Table

- Trials requested: `200`, random seed: `20260313`, blocked edges per trial: `4`
- B1 voxel config: `xy_step=125m`, `agl_step=5m`, timeout `8.0s`

## Per-baseline summary

| Baseline | Success | Replan ms (mean+/-std) | P50/P95 ms | Expanded (mean) | Cost (mean) | Energy kJ (mean) | Length km (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered | 200/200 (100.0%) | 44.10+/-50.86 | 25.75/130.00 | 695.9 | 10.1275 | 253.58 | 6.055 |
| B2_GlobalAstar_Layered | 200/200 (100.0%) | 17.84+/-12.45 | 16.00/44.17 | 1892.5 | 10.1275 | 253.58 | 6.055 |
| B3_LPA_SingleLayer | 200/200 (100.0%) | 5.29+/-28.70 | 0.37/20.41 | 48.4 | 11.4335 | 219.69 | 6.080 |
| B1_Voxel_Dijkstra | 198/200 (99.0%) | 362.11+/-185.29 | 363.52/663.31 | 61705.7 | 60.3283 | 289.92 | 7.218 |

## Paired significance checks

| Pair | Metric | N paired | Mean A | Mean B | Median(B/A) | p-value |
|---|---|---:|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered vs B2_GlobalAstar_Layered | replan_ms | 200 | 44.1013 | 17.8447 | 0.463 | 1.198e-15 |
| B4_Proposed_LPA_Layered vs B3_LPA_SingleLayer | path_cost | 200 | 10.1275 | 11.4335 | 1.134 | 8.901e-75 |
| B4_Proposed_LPA_Layered vs B1_Voxel_Dijkstra | replan_ms | 198 | 43.0369 | 362.1076 | 10.950 | 3.969e-70 |
