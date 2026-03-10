# Benchmark Matrix Table

- Trials per combo: `50`
- n_block grid: `2,4,6,8`
- K grid: `5`
- Scales: `large`
- Seed: `20260310`
- Focus: scale `large`, K-intensity `5`, K-scale `5`, n_block-cont `4`

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 24.22 | 23.66 | 0.977 | 364.80 | 2374.75 | 100.0% | 100.0% |
| 4 | 17.61 | 24.75 | 1.405 | 264.27 | 2480.04 | 100.0% | 100.0% |
| 6 | 23.33 | 26.47 | 1.134 | 353.08 | 2672.86 | 100.0% | 100.0% |
| 8 | 23.48 | 23.44 | 0.998 | 354.46 | 2372.54 | 100.0% | 100.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 88.05 | 123.73 | 1.405 | 1321.34 | 12400.20 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| large | 6749 | 32594 | 88.05 | 123.73 | 1.405 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9022.64 | 19598.22 | 18830.42 | 19593.22 | 892.74 | 0.00 | 1824.00 | 11873.76 |
| 4 | 6382.82 | 20382.60 | 13792.76 | 20377.60 | 644.42 | 0.00 | 1321.34 | 12400.20 |
| 6 | 8652.14 | 21968.66 | 18358.82 | 21963.66 | 865.82 | 0.00 | 1765.38 | 13364.32 |
| 8 | 8717.40 | 19624.12 | 18369.34 | 19619.12 | 865.36 | 0.00 | 1772.32 | 11862.68 |

