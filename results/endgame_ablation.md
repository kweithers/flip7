# End-game curriculum experiment — results

Seeds per arm: 5 ([0, 1, 2, 3, 4]). Units = runs (final checkpoint at ~500k steps).

| metric | step 0 | control mean [95% CI] | curriculum mean [95% CI] | diff | p (two-sided) |
|---|---|---|---|---|---|
| P2 | 0.453 | 0.471 [0.464, 0.478] | 0.470 [0.459, 0.481] | -0.001 | 0.828 |
| P1 | 0.445 | 0.469 [0.458, 0.479] | 0.463 [0.454, 0.471] | -0.006 | 0.246 |
| q_agent | 0.755 | 0.735 [0.726, 0.743] | 0.730 [0.720, 0.739] | -0.005 | 0.310 |
| random | 0.947 | 0.953 [0.952, 0.954] | 0.955 [0.949, 0.960] | +0.001 | 0.494 |
| suite_vs_ppo_best | 0.500 | 0.514 [0.513, 0.515] | 0.518 [0.510, 0.527] | +0.004 | 0.263 |
| h2h_ppo_best | 0.500 | 0.522 [0.515, 0.529] | 0.523 [0.514, 0.533] | +0.001 | 0.825 |
| h2h_ppo_final | 0.630 | 0.645 [0.631, 0.659] | 0.651 [0.644, 0.658] | +0.006 | 0.321 |

- **Primary (P2)**: diff -0.001, Welch p = 0.8285, permutation p = 0.9127
- No-harm checks (diff ≥ −1 pp and not significantly worse): {'P1': True, 'q_agent': True, 'random': True}
- **Success: False**
- Full-game improvement claimed (P1 significant): False
