# End-game experiment — pilot

Runs: control seeds [0, 1, 2, 3, 4], curriculum seeds [0, 1]

| metric | step 0 | control mean ± sd | curriculum mean ± sd |
|---|---|---|---|
| P2 | 0.453 | 0.471 ± 0.006 | 0.476 ± 0.000 |
| P1 | 0.445 | 0.469 ± 0.008 | 0.469 ± 0.005 |
| q_agent | 0.755 | 0.735 ± 0.007 | 0.729 ± 0.000 |
| random | 0.947 | 0.953 ± 0.001 | 0.958 ± 0.001 |
| suite_vs_ppo_best | 0.500 | 0.514 ± 0.001 | 0.525 ± 0.002 |
| h2h_ppo_best | 0.500 | 0.522 ± 0.006 | 0.529 ± 0.007 |
| h2h_ppo_final | 0.630 | 0.645 ± 0.011 | 0.650 ± 0.004 |

- Degradation gate (control P1 below step 0 by > 2 pp): **pass** (control 0.469 vs step 0 0.445)
- P2 seed-to-seed SD (control): 0.0056; upper 80% bound 0.0087
- Seeds per arm for 80% power at 3 pp MDE on P2: **2**
- P1 SD (control): 0.0084
- Curriculum scenario step share: 0.308, 0.308
