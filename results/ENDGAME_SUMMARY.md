# End-game curriculum experiment — summary

Plan: `docs/endgame_curriculum_plan.md`. Detailed outputs: `endgame_headroom.md`,
`endgame_pilot.md`, `endgame_ablation.md`.

## Result: no effect
Fine-tuning `ppo_best` with ~30% of training steps starting from close end-game positions did
**not** improve end-game play over an identical fine-tune with normal starts.

- Primary (P2, win rate from 1,998 real close end-game positions vs ThresholdAgent):
  control 47.1%, curriculum 47.0% (diff −0.1 pp, 95% CIs ±0.7–1.1 pp, p = 0.83).
- Full games vs ThresholdAgent (P1): 46.9% vs 46.3% (p = 0.25). No-harm checks pass.
- 5 seeds per arm; seed-to-seed SD ≈ 0.6 pp, so the experiment could detect ~1.2 pp. The
  pre-registered 3 pp effect is ruled out.

## What did help
Fine-tuning itself (both arms, 500k steps, lr 3e-5, opponents 70% ppo_best / 30% Q-agent):
- vs ThresholdAgent full games: 44.5% → ~46.9%
- head-to-head vs ppo_best: ~52%; vs ppo_final 63% → ~65%
- vs Q-agent: 75.6% → ~73.5% (small regression)

## Deviations from plan
- ThresholdAgent calibration: ppo_best won only 43–50% against it at every aggression tried
  (target window 55–85%). Kept aggression 0.2; it still leaves room in both directions.
- Seeds: sizing called for 2 per arm; ran 5 per arm since runs are cheap (~6 min).
- Code lives in `flip7/experiments/` rather than `flip7/training/`.

## Interpretation
- ppo_best is already strongly score-aware (draws 37 pp more often when trailing than when
  leading in matched situations), which likely explains why extra end-game practice didn't help.
- The bigger signal: a simple one-step expected-value rule (ThresholdAgent) beats ppo_best
  ~57–43. The PPO agent's weakness looks like the basic draw/stay arithmetic, not end-game
  strategy — which points back at adding P(bust)/E(draw) observation features.
