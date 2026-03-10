# Benchmark Matrix Table

- Trials per combo: `50`
- n_block grid: `4`
- K grid: `5`
- Scales: `small,medium,large`
- Seed: `20260310`
- Focus: scale `large`, K-intensity `5`, K-scale `5`, n_block-cont `4`

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 15.59 | 19.08 | 1.224 | 241.06 | 2010.42 | 100.0% | 100.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 77.95 | 95.38 | 1.224 | 1205.28 | 10052.12 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 2066 | 9704 | 36.78 | 37.44 | 1.018 | 100.0% | 100.0% |
| medium | 3893 | 18773 | 55.69 | 68.53 | 1.230 | 100.0% | 100.0% |
| large | 6749 | 32594 | 77.95 | 95.38 | 1.224 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5880.44 | 16689.84 | 12483.84 | 16684.84 | 586.68 | 0.00 | 1205.28 | 10052.12 |

