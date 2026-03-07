# Flip 7 — Project Reference for Claude

## Project Overview

Flip 7 is a two-player push-your-luck card game implemented with:
- A Python reinforcement learning pipeline (Q-learning + PPO via Stable Baselines3)
- A SvelteKit webapp that lets users play against the trained AI in the browser using ONNX Runtime Web

## Repository Structure

```
flip7/
  engine/
    deck.py          # Deck: value v appears v times (v=1..12), 78 cards total
    player.py        # Player state: hand, total_score, busted, stayed
    game.py          # Game loop, Action enum (STAY=0, DRAW=1), GameResult
  agents/
    base.py          # Abstract Agent base class
    random_agent.py  # Random agent with configurable draw_prob
    q_agent.py       # Tabular Q-learning agent
    ppo_agent.py     # SB3 PPO wrapper agent
  environments/
    gym_env.py       # Gymnasium env for PPO training (43-dim obs, Discrete(2))
  training/
    train_q.py       # Q-learning training loop
    train_ppo.py     # PPO training: Stage 1 (vs Q-agent) + Stage 2 (self-play)
    export_onnx.py   # Export PPO policy to ONNX for browser inference
  config.py

models/
  q_table.pkl              # Trained Q-table
  ppo_final.zip            # Final PPO model (SB3 format)
  ppo_best.zip             # Best checkpoint during training
  ppo_best_vecnorm.pkl     # VecNormalize stats for best model
  vec_normalize_final.pkl  # VecNormalize stats for final model
  ppo_flip7.onnx           # ONNX export for browser inference
  normalize_stats.json     # Observation normalization stats (mean/var) for webapp

webapp/
  src/
    lib/
      gameEngine.ts  # TypeScript port of Python game engine
      onnxAgent.ts   # ONNX Runtime Web agent (loads .onnx + normalize_stats.json)
    routes/
      +page.svelte   # Main game UI (Svelte 5 runes)
      +layout.svelte

main.py              # Entry point / CLI
pyproject.toml       # uv-managed Python project
```

## Tech Stack

- **Python**: 3.12, managed with `uv`
- **ML**: PyTorch <2.3 (macOS x86_64 constraint), Stable Baselines3, Gymnasium
- **ONNX**: exported with opset 17; `onnxruntime` only for verification (not macOS x86_64 compatible); browser uses `onnxruntime-web`
- **Webapp**: SvelteKit, Svelte 5 (runes syntax), TypeScript, `npm`
- **Deployment**: `@sveltejs/adapter-vercel`

## Game Rules

- Deck: card value `v` appears `v` times for v=1..12 (78 cards total)
- A player busts if they draw a card they already hold
- Busted players score 0 for the round; otherwise score = hand sum
- Players alternate turns within rounds; first to 200 total wins
- **Both-over-200 rule**: if both players cross 200 in the same round, highest score wins
- Deck persists across rounds; only reshuffled when empty (excluding cards in play)
- Players alternate who goes first each round (`round_number % 2`)

## Observation Space (43 dims)

Used by both `Flip7Env` (training) and `OnnxAgent` (browser):

| Index   | Description                                       |
|---------|---------------------------------------------------|
| 0–11    | My hand presence (binary, has card 1..12)         |
| 12      | My hand size / 12                                 |
| 13      | My hand sum / 78                                  |
| 14      | My total score / 200 (clamped to 1.0)            |
| 15–26   | Opponent hand presence (binary)                   |
| 27      | Opponent hand size / 12                           |
| 28      | Opponent hand sum / 78                            |
| 29      | Opponent total score / 200 (clamped to 1.0)      |
| 30      | Opponent has stayed this round (0 or 1)           |
| 31–42   | Deck counts for values 1..12, normalized by `v`  |

## Reward Function (PPO)

- Game win: +10, game loss: -10
- Round shaping: `(my_round_score - opp_round_score) / 78`
- No bust-specific penalty (keeps agent from being overly conservative)
- Truncation at 50 rounds

## Q-Agent Design

- State: 12-bit card presence tuple `(has_1, has_2, ..., has_12)` — ~1500 reachable states
- Reward: STAY → `hand_sum / 78`, bust → `0` (no negative penalty)
- Trained for 1,000,000 episodes with epsilon-greedy decay

## PPO Training Pipeline

**Stage 1** (`train_stage1`): 2M steps vs Q-agent opponent, 8 parallel envs, VecNormalize
**Stage 2** (`train_stage2`): 1M steps self-play, frozen opponent updated every 200k steps

PPO hyperparameters:
- `learning_rate=3e-4`, `n_steps=2048`, `batch_size=128`, `n_epochs=10`
- `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`
- Network: `pi=[128, 128], vf=[128, 128]`
- TensorBoard logs: `./tb_logs/`

Best model saved automatically when win rate improves (`WinRateCallback`, every 50k steps, 200 eval games).

## ONNX Export

`PolicyWrapper` wraps `mlp_extractor.forward_actor` + `action_net` to produce action logits.
Input: `observation` [batch, 43], Output: `action_logits` [batch, 2].
Decision: `argmax(logits)` → 0=STAY, 1=DRAW.
Normalization stats exported separately as `normalize_stats.json` (mean + var arrays).

## Training Results

| Matchup              | Win Rate |
|----------------------|----------|
| Q-agent vs Random    | 93.8%    |
| PPO vs Random        | 88.7%    |
| PPO vs Q-agent       | 43.8%    |

PPO has not yet beaten Q-agent; needs more training or tuning.

## Common Commands

```bash
# Python — run from project root with uv
uv run python -m flip7.training.train_q
uv run python -m flip7.training.train_ppo
uv run python -m flip7.training.export_onnx

# Webapp — run from webapp/
npm run dev
npm run build
```

## Key Gotchas

- `onnxruntime` does NOT work on macOS x86_64 — only used in browser via `onnxruntime-web`
- The ONNX model must be placed in `webapp/static/ppo_flip7.onnx` and `normalize_stats.json` in `webapp/static/` for the webapp to load them at `/ppo_flip7.onnx` and `/normalize_stats.json`
- Svelte 5 runes syntax is used (`$state`, `$derived`) — not Svelte 4 stores
- The `ppo_agent.py` `_build_obs` must stay in sync with `gym_env.py` `_get_obs` and `onnxAgent.ts` `buildObservation` — all three must produce identical 43-dim vectors
- VecNormalize stats must be re-exported to `normalize_stats.json` any time a new model is trained
