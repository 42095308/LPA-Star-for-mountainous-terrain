# Benchmark Supplement (Median / P95)

## Figure 4 Distribution Stats

| Scenario | Baseline | N | Mean ms | Median ms | P95 ms | Std ms |
|---|---|---:|---:|---:|---:|---:|
| Intensity K=5 | B4_Proposed_LPA_Layered | 50 | 88.05 | 50.35 | 264.50 | 93.11 |
| Intensity K=5 | B2_GlobalAstar_Layered | 50 | 123.73 | 115.07 | 251.21 | 73.95 |
| Scale K=5,n_block=4 | B4_Proposed_LPA_Layered | 50 | 77.95 | 37.72 | 289.13 | 94.93 |
| Scale K=5,n_block=4 | B2_GlobalAstar_Layered | 50 | 95.38 | 90.79 | 207.47 | 61.98 |
| Continuous K=7,n_block=4 | B4_Proposed_LPA_Layered | 50 | 148.56 | 121.45 | 373.35 | 119.71 |
| Continuous K=7,n_block=4 | B2_GlobalAstar_Layered | 50 | 189.30 | 173.35 | 366.36 | 112.63 |

## Optional Long-Horizon Point (K=10, scale=large, n_block=4)

| Baseline | N | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|---:|
| B4_Proposed_LPA_Layered | 50 | 139.46 | 117.56 | 391.08 |
| B2_GlobalAstar_Layered | 50 | 283.59 | 290.35 | 534.87 |
