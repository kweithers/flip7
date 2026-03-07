# Flip 7

A friend introduced me to [Flip 7](https://boardgamegeek.com/boardgame/420087/flip-7), a push-your-luck card game, and I figured it would be a perfect candidate for reinforcement learning. The game has just enough complexity — a shared deck, partial information, and a bust mechanic — to make pure heuristics interesting and learned strategies potentially powerful.

**Play here: [flip7-beta.vercel.app](https://flip7-beta.vercel.app/)**

## Game Rules

The deck contains card values 1–12, where each value `v` appears `v` times (78 cards total). On your turn, you either draw a card or stay. If you draw a card you already hold, you bust and score 0 for the round. Otherwise, your round score is the sum of your hand. First player to 200 total points wins. If both players cross 200 in the same round, the higher score wins.

## Reinforcement Learning Approach

I trained two agents using a curriculum learning strategy: start simple, then build on it.

### Stage 1 — Q-Learning

The first agent uses tabular Q-learning. The state is a 12-bit binary tuple representing which card values the player currently holds (~1,500 reachable states). This is compact enough for a Q-table to converge reliably.

- **Reward**: `hand_sum / 78` on STAY, `0` on bust (no negative penalty — penalizing busts made the agent too conservative)
- **Training**: 1,000,000 episodes with epsilon-greedy decay
- **Result**: 93.8% win rate vs. a random agent

### Stage 2 — PPO with Curriculum Learning

The PPO agent uses a 43-dimensional observation space that includes both players' hands, scores, whether the opponent has stayed, and normalized deck counts. This gives the agent information to reason about risk based on what's left in the deck.

Curriculum learning was key here: rather than training PPO against a random opponent from scratch, training started against the Q-learning agent. This forced the PPO agent to learn meaningful strategies from the beginning instead of exploiting a weak baseline.

- **Stage 1 training**: 2M steps vs. Q-agent opponent, 8 parallel environments with VecNormalize
- **Stage 2 training**: 1M steps of self-play, frozen opponent updated every 200k steps
- **Reward**: game win/loss ±10, round score differential `(my_score - opp_score) / 78` as shaping; episodes truncated at 50 rounds
- **Network**: `pi=[128, 128], vf=[128, 128]`, `lr=3e-4`, `gamma=0.99`, `ent_coef=0.01`
- **Best model**: saved automatically when win rate improves (evaluated every 50k steps over 200 games)
- **Result**: 97.4% win rate vs. random; 70.4% vs. Q-agent

The PPO policy is exported to ONNX and runs entirely in the browser via `onnxruntime-web`. Observation normalization stats (mean/variance) are exported separately as `normalize_stats.json` and applied client-side before inference.