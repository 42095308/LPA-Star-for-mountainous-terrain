# Benchmark Matrix Table

- Trials per combo: `50`
- n_block grid: `2,4,6,8`
- K grid: `3`
- Scales: `large`
- Seed: `20260310`
- Focus: scale `large`, K-intensity `3`, K-scale `3`, n_block-cont `4`

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 31.34 | 24.29 | 0.775 | 484.42 | 2496.80 | 100.0% | 100.0% |
| 4 | 21.63 | 19.58 | 0.905 | 327.13 | 2007.99 | 100.0% | 100.0% |
| 6 | 27.91 | 22.34 | 0.800 | 428.28 | 2300.75 | 100.0% | 100.0% |
| 8 | 35.11 | 23.90 | 0.681 | 523.45 | 2447.10 | 100.0% | 100.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 64.90 | 58.73 | 0.905 | 981.40 | 6023.98 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| large | 6749 | 32594 | 64.90 | 58.73 | 0.905 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 7315.28 | 12312.58 | 15042.58 | 12309.58 | 714.10 | 0.00 | 1453.26 | 7490.40 |
| 4 | 4866.22 | 10081.78 | 10257.44 | 10078.78 | 471.32 | 0.00 | 981.40 | 6023.98 |
| 6 | 6359.44 | 11391.40 | 13361.00 | 11388.40 | 624.42 | 0.00 | 1284.84 | 6902.24 |
| 8 | 8009.34 | 12091.16 | 16467.68 | 12088.16 | 765.38 | 0.00 | 1570.34 | 7341.30 |

