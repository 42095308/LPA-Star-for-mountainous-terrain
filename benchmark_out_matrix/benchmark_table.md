# Benchmark Matrix Table

- Trials per combo: `50`
- n_block grid: `2,4,6,8`
- K grid: `1,2,3,5,8`
- Scales: `large`
- Seed: `20260310`
- Focus: scale `large`, K-intensity `3`, K-scale `3`, n_block-cont `4`
- Experiment A diagnosis: non-monotonic timing detected at scale=large, K=3. B4 mean event time changes from 24.53ms (n_block=2) to 35.82ms (n_block=8), while expanded nodes move from 400.1 to 564.3. This indicates wall-clock is jointly affected by event geometry/locality and Python runtime overhead, not only by n_block magnitude. A typical case is that more blocked edges can force an earlier detour into a cleaner subgraph, which shortens the actually affected middle segment and reduces updated states.
- Experiment B diagnosis: for K=1 (scale=large, n_block=4), B4 is slower than B2 (B2/B4=0.391<1). This is expected under a single light perturbation: incremental LPA* still pays queue/state-maintenance overhead, while global A* can finish quickly when the affected region is tiny. As K increases, LPA* reuses prior search state and cumulative advantage emerges.
- Path-quality diagnosis: B4/B2 path costs are frequently equal (mean equal-cost rate=100.0%). This is expected because both optimize the same weighted objective on the same graph; incremental LPA* mainly improves replanning workload/time rather than final optimality.

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 24.53 | 19.63 | 0.800 | 400.14 | 2086.39 | 100.0% | 100.0% |
| 4 | 29.55 | 20.80 | 0.704 | 468.11 | 2157.01 | 100.0% | 100.0% |
| 6 | 26.33 | 17.48 | 0.664 | 416.63 | 1831.22 | 100.0% | 100.0% |
| 8 | 35.82 | 22.00 | 0.614 | 564.27 | 2281.35 | 100.0% | 100.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55.66 | 21.74 | 0.391 | 886.28 | 2235.78 | 100.0% | 100.0% |
| 2 | 61.23 | 32.44 | 0.530 | 973.16 | 3418.22 | 100.0% | 100.0% |
| 3 | 88.65 | 62.39 | 0.704 | 1404.34 | 6471.02 | 100.0% | 100.0% |
| 5 | 69.86 | 86.30 | 1.235 | 1115.54 | 9250.94 | 100.0% | 100.0% |
| 8 | 106.47 | 169.20 | 1.589 | 1717.70 | 18424.42 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| large | 6749 | 31886 | 88.65 | 62.39 | 0.704 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5761.48 | 10798.66 | 12090.04 | 10795.66 | 578.66 | 0.00 | 1200.42 | 6259.18 |
| 4 | 6831.30 | 11152.06 | 14174.98 | 11149.06 | 683.38 | 0.00 | 1404.34 | 6471.02 |
| 6 | 6047.48 | 9469.76 | 12678.14 | 9466.76 | 603.80 | 0.00 | 1249.88 | 5493.66 |
| 8 | 8522.84 | 11804.58 | 17109.90 | 11801.58 | 825.68 | 0.00 | 1692.82 | 6844.06 |

