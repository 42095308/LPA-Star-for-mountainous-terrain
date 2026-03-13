# Benchmark Matrix Table

- Trials per combo: `100`
- n_block grid: `2,4,6,8`
- K grid: `1,3,5,7,10`
- Scales: `small,medium,large`
- Seed: `20260313`
- Focus: scale `large`, K-intensity `5`, K-scale `5`, n_block-cont `4`

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 28.18 | 30.31 | 1.075 | 312.32 | 2151.49 | 100.0% | 100.0% |
| 4 | 19.09 | 21.05 | 1.102 | 297.35 | 2194.58 | 100.0% | 100.0% |
| 6 | 17.34 | 18.34 | 1.058 | 271.95 | 1930.52 | 100.0% | 100.0% |
| 8 | 17.19 | 17.27 | 1.004 | 267.12 | 1808.89 | 99.0% | 99.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.48 | 20.06 | 0.397 | 761.60 | 1973.44 | 100.0% | 100.0% |
| 3 | 82.70 | 54.07 | 0.654 | 1280.47 | 5517.76 | 100.0% | 100.0% |
| 5 | 95.46 | 105.23 | 1.102 | 1486.77 | 10972.89 | 100.0% | 100.0% |
| 7 | 92.26 | 124.04 | 1.344 | 1455.24 | 13043.76 | 100.0% | 100.0% |
| 10 | 93.10 | 168.13 | 1.806 | 1451.78 | 18243.59 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 2066 | 9704 | 30.10 | 33.46 | 1.112 | 100.0% | 100.0% |
| medium | 3893 | 18773 | 49.04 | 60.96 | 1.243 | 100.0% | 100.0% |
| large | 6749 | 32594 | 95.46 | 105.23 | 1.102 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 7546.55 | 19369.24 | 16138.69 | 19364.24 | 761.98 | 0.00 | 1561.60 | 10757.43 |
| 4 | 7173.42 | 19827.04 | 15534.23 | 19822.04 | 724.45 | 0.00 | 1486.77 | 10972.89 |
| 6 | 6522.79 | 17327.13 | 14125.37 | 17322.13 | 662.40 | 0.00 | 1359.73 | 9652.59 |
| 8 | 6419.30 | 16338.96 | 13953.91 | 16333.96 | 652.10 | 0.00 | 1335.61 | 9044.44 |

