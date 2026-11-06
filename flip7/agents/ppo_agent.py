"""PPO agent wrapper for playing games with the Agent interface."""

import numpy as np
from stable_baselines3 import PPO

from flip7.agents.base import Agent
from flip7.engine.features import build_obs
from flip7.engine.game import Action, Game
from flip7.engine.player import Player


class PPOAgent(Agent):
    """Wraps a trained SB3 PPO model to play via the Agent interface."""

    def __init__(self, model: PPO, obs_rms=None, obs_variant: str = "base43"):
        self.model = model
        self.obs_variant = obs_variant
        self.obs_rms = obs_rms  # VecNormalize stats if used

    def _build_obs(self, player: Player, opponent: Player, game: Game) -> np.ndarray:
        """Build the observation vector for this agent's variant (default 43-dim)."""
        return build_obs(player, opponent, game, self.obs_variant)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is not None:
            obs = (obs - self.obs_rms["mean"]) / np.sqrt(
                self.obs_rms["var"] + 1e-8
            )
            obs = np.clip(obs, -10.0, 10.0)
        return obs

    def choose_action(self, player: Player, opponent: Player, game: Game) -> Action:
        obs = self._build_obs(player, opponent, game)
        obs = self._normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return Action(int(action))

    @classmethod
    def load(cls, model_path: str, stats_path: str | None = None) -> "PPOAgent":
        model = PPO.load(model_path)
        obs_rms = None
        if stats_path:
            import json

            with open(stats_path) as f:
                obs_rms = json.load(f)
            obs_rms["mean"] = np.array(obs_rms["mean"], dtype=np.float32)
            obs_rms["var"] = np.array(obs_rms["var"], dtype=np.float32)
        return cls(model, obs_rms)
