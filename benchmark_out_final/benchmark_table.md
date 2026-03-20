# Benchmark Matrix Table

- Trials per combo: `50`
- n_block grid: `2,4,6,8`
- K grid: `1,3,5,7,10`
- Scales: `small,medium,large`
- Seed: `20260309`
- Focus: scale `large`, K-intensity `5`, K-scale `5`, n_block-cont `4`
- Experiment A diagnosis: non-monotonic timing detected at scale=large, K=5. B4 mean event time changes from 19.72ms (n_block=2) to 17.82ms (n_block=8), while expanded nodes move from 315.6 to 284.8. This indicates wall-clock is jointly affected by event geometry/locality and Python runtime overhead, not only by n_block magnitude. A typical case is that more blocked edges can force an earlier detour into a cleaner subgraph, which shortens the actually affected middle segment and reduces updated states.
- Experiment B diagnosis: for K=1 (scale=large, n_block=4), B4 is slower than B2 (B2/B4=0.330<1). This is expected under a single light perturbation: incremental LPA* still pays queue/state-maintenance overhead, while global A* can finish quickly when the affected region is tiny. As K increases, LPA* reuses prior search state and cumulative advantage emerges.
- Path-quality diagnosis: B4/B2 path costs are frequently equal (mean equal-cost rate=100.0%). This is expected because both optimize the same weighted objective on the same graph; incremental LPA* mainly improves replanning workload/time rather than final optimality.

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 19.72 | 17.52 | 0.888 | 315.58 | 1936.00 | 100.0% | 100.0% |
| 4 | 18.29 | 17.76 | 0.971 | 293.45 | 1994.26 | 100.0% | 100.0% |
| 6 | 18.28 | 18.59 | 1.017 | 295.49 | 2078.77 | 100.0% | 100.0% |
| 8 | 17.82 | 17.99 | 1.010 | 284.78 | 2011.51 | 100.0% | 100.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 69.86 | 23.07 | 0.330 | 1098.08 | 2385.66 | 100.0% | 100.0% |
| 3 | 82.93 | 58.69 | 0.708 | 1339.08 | 6458.96 | 100.0% | 100.0% |
| 5 | 91.45 | 88.82 | 0.971 | 1467.26 | 9971.30 | 100.0% | 100.0% |
| 7 | 86.04 | 123.52 | 1.436 | 1395.84 | 13934.86 | 100.0% | 100.0% |
| 10 | 126.50 | 206.01 | 1.629 | 2058.26 | 23677.98 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 2066 | 9366 | 23.76 | 31.15 | 1.311 | 100.0% | 100.0% |
| medium | 3892 | 18117 | 47.00 | 58.58 | 1.246 | 100.0% | 100.0% |
| large | 6749 | 31886 | 91.45 | 88.82 | 0.971 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 7553.28 | 16790.00 | 15940.94 | 16785.00 | 769.10 | 0.00 | 1577.90 | 9679.98 |
| 4 | 7030.96 | 17317.84 | 14811.38 | 17312.84 | 706.04 | 0.00 | 1467.26 | 9971.30 |
| 6 | 6966.08 | 17842.28 | 14946.12 | 17837.28 | 720.92 | 0.00 | 1477.46 | 10393.84 |
| 8 | 6888.02 | 17293.58 | 14473.60 | 17288.58 | 690.10 | 0.00 | 1423.90 | 10057.56 |

