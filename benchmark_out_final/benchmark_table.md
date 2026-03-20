# Benchmark Matrix Table

- Trials per combo: `200`
- n_block grid: `2,4,6,8`
- K grid: `1,3,5,7,10`
- Scales: `small,medium,large`
- Seed: `20260313`
- Focus: scale `large`, K-intensity `5`, K-scale `5`, n_block-cont `4`
- Experiment A diagnosis: non-monotonic timing detected at scale=large, K=5. B4 mean event time changes from 18.59ms (n_block=2) to 20.80ms (n_block=8), while expanded nodes move from 298.4 to 328.7. This indicates wall-clock is jointly affected by event geometry/locality and Python runtime overhead, not only by n_block magnitude. A typical case is that more blocked edges can force an earlier detour into a cleaner subgraph, which shortens the actually affected middle segment and reduces updated states.
- Experiment B diagnosis: for K=1 (scale=large, n_block=4), B4 is slower than B2 (B2/B4=0.372<1). This is expected under a single light perturbation: incremental LPA* still pays queue/state-maintenance overhead, while global A* can finish quickly when the affected region is tiny. As K increases, LPA* reuses prior search state and cumulative advantage emerges.
- Path-quality diagnosis: B4/B2 path costs are frequently equal (mean equal-cost rate=100.0%). This is expected because both optimize the same weighted objective on the same graph; incremental LPA* mainly improves replanning workload/time rather than final optimality.

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 18.59 | 19.82 | 1.066 | 298.36 | 2134.80 | 100.0% | 100.0% |
| 4 | 20.08 | 20.52 | 1.022 | 320.13 | 2190.16 | 100.0% | 100.0% |
| 6 | 20.20 | 20.43 | 1.011 | 321.83 | 2177.90 | 100.0% | 100.0% |
| 8 | 20.80 | 18.60 | 0.894 | 328.69 | 1988.81 | 100.0% | 100.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56.30 | 20.92 | 0.372 | 880.00 | 2106.55 | 100.0% | 100.0% |
| 3 | 84.58 | 60.45 | 0.715 | 1337.58 | 6289.14 | 100.0% | 100.0% |
| 5 | 100.42 | 102.59 | 1.022 | 1600.63 | 10950.82 | 100.0% | 100.0% |
| 7 | 101.01 | 136.69 | 1.353 | 1610.57 | 14827.17 | 99.5% | 99.5% |
| 10 | 111.70 | 185.98 | 1.665 | 1796.23 | 20528.12 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 2066 | 9366 | 27.37 | 31.41 | 1.148 | 100.0% | 100.0% |
| medium | 3892 | 18117 | 81.26 | 84.66 | 1.042 | 100.0% | 100.0% |
| large | 6749 | 31886 | 100.42 | 102.59 | 1.022 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 7112.40 | 18432.90 | 15036.89 | 18427.90 | 726.21 | 0.00 | 1491.79 | 10674.01 |
| 4 | 7617.48 | 18882.35 | 16148.06 | 18877.35 | 779.12 | 0.00 | 1600.63 | 10950.82 |
| 6 | 7659.31 | 18762.97 | 16250.57 | 18757.97 | 782.87 | 0.00 | 1609.17 | 10889.52 |
| 8 | 7955.40 | 17192.77 | 16638.11 | 17187.77 | 800.90 | 0.00 | 1643.43 | 9944.03 |

