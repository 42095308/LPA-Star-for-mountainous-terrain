# Benchmark Discussion Notes

- Experiment A: non-monotonic timing detected at scale=large, K=5. B4 mean event time changes from 20.56ms (n_block=2) to 29.84ms (n_block=8), while expanded nodes move from 315.6 to 284.8. This indicates wall-clock is jointly affected by event geometry/locality and Python runtime overhead, not only by n_block magnitude. A typical case is that more blocked edges can force an earlier detour into a cleaner subgraph, which shortens the actually affected middle segment and reduces updated states.
- Experiment B: for K=1 (scale=large, n_block=4), B4 is slower than B2 (B2/B4=0.347<1). This is expected under a single light perturbation: incremental LPA* still pays queue/state-maintenance overhead, while global A* can finish quickly when the affected region is tiny. As K increases, LPA* reuses prior search state and cumulative advantage emerges.
- Path quality: B4/B2 path costs are frequently equal (mean equal-cost rate=100.0%). This is expected because both optimize the same weighted objective on the same graph; incremental LPA* mainly improves replanning workload/time rather than final optimality.

Suggested paper wording:
1. Under K=1 and light perturbation, incremental LPA* may be slower due to queue/state-maintenance overhead.
2. With larger K, state reuse accumulates and cumulative advantage becomes clear.
3. Non-monotonic time vs n_block can happen when event geometry pushes detours into cleaner subgraphs, reducing updated states.
4. Equal B4/B2 path costs are expected for many trials because both solve the same weighted objective; speed/workload is the key differentiator.
