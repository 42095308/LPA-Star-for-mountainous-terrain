# Benchmark Matrix Table

- Trials per combo: `100`
- n_block grid: `2,4,6,8`
- K grid: `1,3,5`
- Scales: `small,medium,large`
- Seed: `20260310`
- Focus: scale `large`, K-intensity `3`, K-scale `3`, n_block-cont `4`

## Experiment A (Event Intensity)

| n_block | B4 event ms | B2 event ms | B2/B4 | B4 event expanded | B2 event expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 46.32 | 30.14 | 0.651 | 516.94 | 2390.71 | 100.0% | 100.0% |
| 4 | 33.47 | 21.91 | 0.654 | 514.20 | 2367.19 | 100.0% | 100.0% |
| 6 | 31.73 | 24.31 | 0.766 | 489.55 | 2606.74 | 99.0% | 99.0% |
| 8 | 28.21 | 20.78 | 0.737 | 443.29 | 2268.65 | 99.0% | 99.0% |

## Experiment B (Continuous Replanning)

| K events | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 cumulative expanded | B2 cumulative expanded | B4 success | B2 success |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60.42 | 24.80 | 0.410 | 921.48 | 2582.70 | 100.0% | 100.0% |
| 3 | 100.41 | 65.72 | 0.654 | 1542.59 | 7101.58 | 100.0% | 100.0% |
| 5 | 103.88 | 116.51 | 1.122 | 1622.34 | 12852.13 | 100.0% | 100.0% |

## Experiment C (Scale Sweep)

| Scale | |V| | |E| | B4 cumulative ms | B2 cumulative ms | B2/B4 | B4 success | B2 success |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 2066 | 9704 | 25.91 | 19.61 | 0.757 | 100.0% | 100.0% |
| medium | 3893 | 18773 | 53.30 | 39.94 | 0.749 | 100.0% | 100.0% |
| large | 6749 | 32594 | 100.41 | 65.72 | 0.654 | 100.0% | 100.0% |

## Experiment D (Workload Metrics)

| n_block | B4 queue push | B2 queue push | B4 updated | B2 updated | B4 reopened | B2 reopened | B4 expanded | B2 expanded |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 7794.23 | 11751.18 | 16116.98 | 11748.18 | 761.68 | 0.00 | 1550.81 | 7172.13 |
| 4 | 7795.51 | 11721.69 | 16058.02 | 11718.69 | 750.62 | 0.00 | 1542.59 | 7101.58 |
| 6 | 7325.69 | 12894.07 | 15312.83 | 12891.07 | 716.06 | 0.00 | 1468.65 | 7820.23 |
| 8 | 6615.03 | 11219.37 | 13771.00 | 11216.37 | 647.20 | 0.00 | 1329.88 | 6805.95 |

