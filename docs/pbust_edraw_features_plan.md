# Plan: P(bust) / E(draw) observation features

## Context
The end-game curriculum experiment (`results/ENDGAME_SUMMARY.md`) found no effect, but showed that
a one-step expected-value rule (`ThresholdAgent`) beats every PPO model (~57–43 vs `ppo_best`).
`ppo_best` makes clear draw/stay mistakes on 16% of decisions, getting worse as the deck runs
down (43% wrong draws when < 20 cards remain) — i.e. it can't do the card-counting arithmetic.
Hypothesis: giving the network P(bust) and E(draw) directly as inputs fixes this and improves
win rate. Secondary question: with those summaries available, can a much smaller input set do
as well?

Branch: `claude/pbust-edraw-features` (from `claude/dazzling-newton-n1kate`, reusing its
evaluation code).

## Arms (all trained from scratch, identical recipe and seeds)
| Arm | Inputs | Dims |
|---|---|---|
| **A: baseline** | current 43 | 43 |
| **B: added** | 43 + my P(bust), my E(draw)/78, cards left in deck/78 | 46 |
| **C: compact** | me: P(bust), E(draw)/78, hand sum/78, hand size/12, total/200; opponent: P(bust), E(draw)/78, hand sum/78, total/200, stayed; shared: cards left/78, (my total − opp total)/200 | 12 |

Feature definitions reuse `ThresholdAgent`'s code exactly: `draw_distribution` (handles an empty
deck via the post-reshuffle distribution) and `draw_stats` / `expected_draw_gain`. E(draw)/78 is
clipped to [−1, 1]. Opponent P(bust)/E(draw) use the opponent's hand against the same deck.

## Training recipe
- Original Stage 1 recipe (`train_stage1`): lr 1e-4, n_steps 2048, batch 64, 10 epochs,
  γ 0.99, λ 0.95, clip 0.2, ent 0.01, pi/vf [128, 128], 4 envs, VecNormalize (obs only,
  updating as in the original).
- **Opponent: Q-agent only** (as in Stage 1). Keeps `ThresholdAgent` and `ppo_best` fully held
  out and makes arm A a reproduction of how `ppo_best` was made.
- 2M steps per run (`ppo_best` peaked early in its 5M-step run). 5 seeds per arm (15 runs,
  ~25 min each, 4 in parallel → ~2 h). Seeding as before (`set_random_seed(seed * 1000)`).
- Checkpoint every 250k steps (8 per run), **each saved with its own VecNormalize stats**
  (stats change during training, so a checkpoint is only usable with matching stats).
- Outputs under `models/feat/{arm}_s{seed}/`; existing models never written.

## Evaluation
All against frozen opponents, paired seat-swapped games, common seeds across models.
- **Validation** (checkpoint selection only): 500 seeds × 2 vs `ThresholdAgent`, every checkpoint
  → learning curves, and each run's selected checkpoint.
- **Test** (all reported numbers), on disjoint seeds, for each run's **final** (2M) and
  **selected** checkpoint:
  - **P1 (primary)**: full games vs `ThresholdAgent`, 2,500 seeds × 2.
  - **Mistake rate (diagnostic)**: share of decisions going against a clear one-step EV
    (|E(draw)| > 3 points), by deck size (< 20, 20–40, 40+), from the P1 games.
  - Secondary: P2 (end-game position suite), head-to-head vs `ppo_best`, vs Q-agent, vs Random.
- **Round-robin**: selected model of the best run per arm (by validation), `ppo_best`,
  `ThresholdAgent` — every pair, 1,000 seeds × 2.

## Decision rules (pre-registered)
Units = runs (5 per arm). Primary analysis on **final** checkpoints; selected checkpoints
reported alongside.
- **B vs A (primary)**: features help iff B > A on P1, Welch p < 0.05.
- **C vs B (exploratory)**: C "as good" iff the 95% CI of (C − B) on P1 has lower bound > −1 pp.
- **No-harm**: B (and C if deployed) not worse than A by > 1 pp, nor significantly worse, vs
  Q-agent and Random.
- **Sanity**: arm A should land near `ppo_best` (≈ 44% vs ThresholdAgent). If it's far off,
  stop and find the setup difference before interpreting B/C.

| Outcome | Action |
|---|---|
| B > A, C ≈ B | Deploy C (simpler) |
| B > A, C < B | Deploy B |
| B ≈ A | Features don't help; report and stop |

Any deployed model must also beat `ppo_best` head-to-head (run-level CI above 50%).

## Implementation
1. `flip7/engine/features.py`: `build_obs(player, opponent, game, variant)` for variants
   `base43`, `added46`, `compact12`; `OBS_DIMS`. Move/reuse `draw_distribution`, `draw_stats`,
   `expected_draw_gain` (ThresholdAgent imports them; behaviour unchanged).
2. `Flip7Env(obs_variant="base43")` and `PPOAgent(obs_variant="base43")` both call
   `build_obs` — one Python source of truth. Default stays 43-dim, byte-identical.
3. `flip7/experiments/features.py`: train (`--arm A|B|C --seed N`), per-checkpoint stats
   callback, validation sweep, test eval, report (`results/features_ablation.md`), round-robin.
   Reuse `paired_eval`, `suite_eval`, stats helpers from `evaluate_endgame.py`.
4. `FastPPOAgent` / `build_frozen_ppo` take `obs_variant`.
5. Deployment only if a model wins: `export_onnx` takes input dim + `_load_obs_rms` stats;
   port `build_obs` for the chosen variant to `webapp/src/lib/onnxAgent.ts`, including the
   P(bust)/E(draw) reshuffle rule; parity test against Python on fixed states.

## Tests
- Default env/agent obs identical to current (byte-identical rollout, as before).
- `build_obs` dims per variant; values in range; P(bust)/E(draw) equal ThresholdAgent's
  functions on random states, including empty deck.
- Env and PPOAgent produce the same vector for the same state, per variant.
- ThresholdAgent decisions unchanged after the refactor (fixed seeds).
- Checkpoint callback writes matching stats; training never writes outside `models/feat/`.

## Verification
1. `uv run --extra dev pytest`.
2. Smoke: 20k steps per arm.
3. Full runs → validation sweep → test eval → `results/features_ablation.md` + round-robin.
4. Commit code and results to `claude/pbust-edraw-features` (no model binaries unless deploying).
