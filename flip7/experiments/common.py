"""Shared helpers for the end-game curriculum experiment."""

import numpy as np
import torch
from stable_baselines3 import PPO

from flip7.agents.ppo_agent import PPOAgent
from flip7.engine.game import Action, Game
from flip7.engine.player import Player
from flip7.training.train_ppo import _load_obs_rms

WIN = Game.WINNING_SCORE
CLOSE_MIN = 150
CLOSE_GAP = 40


def is_close_endgame(scores) -> bool:
    """Both scores in [150, 199] and within 40 points of each other."""
    a, b = scores
    return (
        CLOSE_MIN <= a < WIN
        and CLOSE_MIN <= b < WIN
        and abs(a - b) <= CLOSE_GAP
    )


class FastPPOAgent(PPOAgent):
    """PPOAgent with a direct torch forward pass (same argmax as predict)."""

    def __init__(self, model: PPO, obs_rms=None, obs_variant: str = "base43"):
        super().__init__(model, obs_rms, obs_variant)
        self.policy = model.policy
        self.policy.set_training_mode(False)

    def choose_action(self, player: Player, opponent: Player, game: Game) -> Action:
        obs = self._normalize_obs(self._build_obs(player, opponent, game))
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.policy.action_net(self.policy.mlp_extractor.forward_actor(x))
        return Action(int(logits.argmax(dim=1).item()))


def load_obs_stats(vecnorm_path: str) -> dict:
    rms = _load_obs_rms(vecnorm_path)
    return {"mean": rms.mean.astype(np.float32), "var": rms.var.astype(np.float32)}


def build_frozen_ppo(model_path: str, vecnorm_path: str, fast: bool = True,
                     obs_variant: str = "base43") -> PPOAgent:
    """Load a PPO model + its obs stats (custom dict or VecNormalize pickle)."""
    model = PPO.load(model_path, device="cpu")
    cls = FastPPOAgent if fast else PPOAgent
    return cls(model, load_obs_stats(vecnorm_path), obs_variant)


def play(agents, seed: int):
    """Play one full game; returns (GameResult, Game)."""
    game = Game(seed=seed)
    return game.play_game(agents), game


def paired_eval(candidate, opponent, seeds) -> dict:
    """Each seed played twice, candidate in seat 0 then seat 1."""
    wins = games = 0
    for s in seeds:
        for seat in (0, 1):
            agents = [candidate, opponent] if seat == 0 else [opponent, candidate]
            result, _ = play(agents, s)
            wins += int(result.winner == seat)
            games += 1
    return {"wins": wins, "games": games, "win_rate": wins / games}
