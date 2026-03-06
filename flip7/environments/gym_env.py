"""Gymnasium environment for Flip 7 PPO training.

Observation space (41 dims):
  - my_hand_presence (12): binary, do I hold value v?
  - my_hand_size (1): cards / 12
  - my_total_score (1): score / 200
  - opp_hand_presence (12): binary, does opponent hold v?
  - opp_hand_size (1): cards / 12
  - opp_total_score (1): score / 200
  - opp_has_stayed (1): 1 if opponent stayed this round
  - deck_counts (12): count of each value remaining / max count for that value

Action space: Discrete(2) — 0=STAY, 1=DRAW
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flip7.agents.base import Agent
from flip7.engine.game import Action, Game
from flip7.engine.player import Player


class Flip7Env(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    PLAYER_IDX = 0
    OPPONENT_IDX = 1

    def __init__(self, opponent_agent: Agent, render_mode=None):
        super().__init__()
        self.opponent_agent = opponent_agent
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(41,), dtype=np.float32
        )

        self.game: Game | None = None

    def _get_obs(self) -> np.ndarray:
        player = self.game.players[self.PLAYER_IDX]
        opponent = self.game.players[self.OPPONENT_IDX]
        deck_counts = self.game.deck.card_counts()

        obs = np.zeros(41, dtype=np.float32)

        # My hand presence (12)
        for card in player.hand:
            obs[card - 1] = 1.0

        # My hand size (1)
        obs[12] = len(player.hand) / 12.0

        # My total score (1)
        obs[13] = min(player.total_score / 200.0, 1.0)

        # Opponent hand presence (12)
        for card in opponent.hand:
            obs[14 + card - 1] = 1.0

        # Opponent hand size (1)
        obs[26] = len(opponent.hand) / 12.0

        # Opponent total score (1)
        obs[27] = min(opponent.total_score / 200.0, 1.0)

        # Opponent has stayed (1)
        obs[28] = 1.0 if opponent.stayed else 0.0

        # Deck counts (12)
        for v in range(1, 13):
            obs[29 + v - 1] = deck_counts.get(v, 0) / v  # normalize by max count

        return obs

    def _compute_reward(self, busted: bool, round_over: bool, game_over: bool) -> float:
        player = self.game.players[self.PLAYER_IDX]
        opponent = self.game.players[self.OPPONENT_IDX]
        reward = 0.0

        if game_over:
            if self.game.result.winner == self.PLAYER_IDX:
                reward += 10.0
            else:
                reward += -10.0

        if round_over:
            my_round = player.round_score()
            opp_round = opponent.round_score()
            score_diff = (my_round - opp_round) / 78.0
            position = 0.1 * (player.total_score - opponent.total_score) / 200.0
            base = 0.1 if not player.busted else 0.0
            reward += base + score_diff + position

        if busted and not round_over:
            gap = (player.total_score - opponent.total_score) / 200.0
            positional_factor = 1.0 + max(0.0, gap)
            cards_factor = max(0.3, 1.0 - len(player.hand) / 10.0)
            reward += -0.5 * positional_factor * cards_factor

        return reward

    def _run_opponent_turns(self):
        """Run the opponent's turns until they stay/bust or round ends."""
        while (
            self.game.current_player_idx == self.OPPONENT_IDX
            and not self.game.is_round_over()
        ):
            opponent = self.game.players[self.OPPONENT_IDX]
            player = self.game.players[self.PLAYER_IDX]
            action = self.opponent_agent.choose_action(opponent, player, self.game)
            self.game.step(action)

    def _ensure_player_turn(self):
        """After a round ends, start new rounds until it's the player's turn
        or the game is over. Handles opponent-first rounds automatically."""
        while not self.game.game_over:
            self.game.current_player_idx = self.game.round_number % 2
            self.game.start_round()

            if self.game.current_player_idx == self.OPPONENT_IDX:
                self._run_opponent_turns()

            if self.game.is_round_over():
                # Opponent finished the round on their own (edge case)
                # _end_round was already called by game.step() if both done
                # But if opponent was alone and finished, we need to handle it
                if not self.game.game_over:
                    continue
                return
            else:
                # Player's turn
                return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = self.np_random.integers(0, 2**31)
        self.game = Game(seed=int(game_seed))
        self._ensure_player_turn()
        return self._get_obs(), {}

    def step(self, action):
        action = Action(action)
        player = self.game.players[self.PLAYER_IDX]

        if self.game.game_over:
            reward = 10.0 if self.game.result.winner == self.PLAYER_IDX else -10.0
            return self._get_obs(), reward, True, False, self._info()

        # Take our action
        busted = False
        round_over = self.game.step(action)
        busted = player.busted

        if not round_over:
            # Let opponent take their turns
            self._run_opponent_turns()
            round_over = self.game.is_round_over()

        game_over = self.game.game_over

        # Compute reward based on what happened this step
        reward = self._compute_reward(busted, round_over, game_over)

        # If round is over but game isn't, set up for next round
        if round_over and not game_over:
            self._ensure_player_turn()
            game_over = self.game.game_over
            if game_over:
                # Game ended during opponent-only rounds
                if self.game.result.winner == self.PLAYER_IDX:
                    reward += 10.0
                else:
                    reward += -10.0

        terminated = game_over
        truncated = self.game.round_number > 50

        return self._get_obs(), reward, terminated, truncated, self._info()

    def _info(self):
        return {
            "round": self.game.round_number,
            "my_score": self.game.players[self.PLAYER_IDX].total_score,
            "opp_score": self.game.players[self.OPPONENT_IDX].total_score,
        }
