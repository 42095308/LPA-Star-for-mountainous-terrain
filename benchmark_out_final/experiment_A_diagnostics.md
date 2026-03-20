# Experiment A Diagnosis

- non-monotonic timing detected at scale=large, K=5. B4 mean event time changes from 18.59ms (n_block=2) to 20.80ms (n_block=8), while expanded nodes move from 298.4 to 328.7. This indicates wall-clock is jointly affected by event geometry/locality and Python runtime overhead, not only by n_block magnitude. A typical case is that more blocked edges can force an earlier detour into a cleaner subgraph, which shortens the actually affected middle segment and reduces updated states.
