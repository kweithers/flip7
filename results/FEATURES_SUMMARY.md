# P(bust)/E(draw) features — summary

Full tables: `features_ablation.md`. Curves: `features_learning_curves.png`.
Plan: `docs/pbust_edraw_features_plan.md`.

## Result
Both pre-registered tests passed, using final checkpoints, 5 runs per arm.

- **Features help.**
  - B (43 + P(bust), E(draw), cards left) beats A (current 43) on full games vs ThresholdAgent: 44.8% vs 42.6%, +2.2 pp, Welch p = 0.003.
- **The compact set is better, not just as good.**
  - C (12 inputs) scores 47.1% on full games vs ThresholdAgent.
  - C − B = +2.3 pp, 95% CI [+0.8, +3.8] pp.
  - C is also the best arm on every other metric:

    | metric | C | ppo_best |
    |---|---|---|
    | vs ppo_best head-to-head | 54.5% (CI 53.7–55.4) | — |
    | vs Q-agent | 77.1% | 75.5% |
    | vs Random | 96.4% | — |
    | clear-decision mistake rate | 5.0% | 9.1% |

- **No-harm checks pass** for B and C against the Q-agent and Random.
- **Pre-registered outcome:** "B > A, C ≈ B" → deploy C. C also meets the deployment bar of beating ppo_best head-to-head (run-level CI above 50%).

## Notes
- **Sanity check on arm A.** Final checkpoints are 2 pp below ppo_best (42.6% vs 44.5%). Selected checkpoints match it (45.0%).
  - The gap is the late-training decline already noted in CLAUDE.md.
  - So arm A reproduces how ppo_best was made.
- **Learning curves.**
  - Against the held-out ThresholdAgent, win rate declines with training for A and B. It is flat-to-declining for C.
  - Against the training opponent (Q-agent), C keeps improving and A/B are flat.
  - Interpretation: training only against the Q-agent pulls the policy toward exploiting the Q-agent, not toward generally better play.
  - Every arm's best validation point is early (250k–750k for most runs). The true peak may be before the first checkpoint.
- **The features fix the card-counting errors.** Wrong draws when < 20 cards remain:

  | arm | wrong draws |
  |---|---|
  | A | 40% |
  | B | 24% |
  | C | 28% |

  C nearly eliminates mistakes with 40+ cards left (1.6% / 1.0%).
- **Round-robin (best run per arm, selected checkpoint).** The C model beats A, B and ppo_best, and ties ThresholdAgent (49.7%). It is the first PPO model that doesn't lose to it.

## Not done (needs a decision)
- Deploying C means work across both the Python and webapp sides:
  - a 12-input ONNX export;
  - a TypeScript port of the 12 features, including the empty-deck reshuffle rule for P(bust)/E(draw);
  - a Python↔TS parity test;
  - new `normalize_stats.json`.
- Candidate model: C_s3 `ckpt_250000` (best validation score, round-robin entrant), or C_s0 `final`.
