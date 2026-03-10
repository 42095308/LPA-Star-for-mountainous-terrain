# Benchmark Matrix Table

- Trials per combo: `50`
- n_block grid: `4`
- K grid: `7,10`
- Scales: `large`
- Seed: `20260310`
- Focus: scale `large`, K-intensity `7`, K-scale `7`, n_block-cont `4`

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 21.22 | 27.04 | 1.274 | 328.77 | 2805.83 | 100.0% | 100.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 148.56 | 189.30 | 1.274 | 2301.38 | 19640.80 | 100.0% | 100.0% |
| 10 | 139.46 | 283.59 | 2.033 | 2161.40 | 29384.88 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| large | 6749 | 32594 | 148.56 | 189.30 | 1.274 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 11387.56 | 32195.18 | 23834.36 | 32188.18 | 1133.24 | 0.00 | 2301.38 | 19640.80 |

