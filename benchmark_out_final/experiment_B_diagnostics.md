# Experiment B Diagnosis

- for K=1 (scale=large, n_block=4), B4 is slower than B2 (B2/B4=0.372<1). This is expected under a single light perturbation: incremental LPA* still pays queue/state-maintenance overhead, while global A* can finish quickly when the affected region is tiny. As K increases, LPA* reuses prior search state and cumulative advantage emerges.
