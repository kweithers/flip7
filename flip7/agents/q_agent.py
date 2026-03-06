import pickle
import random
from collections import defaultdict

import numpy as np

from flip7.agents.base import Agent
from flip7.engine.game import Action, Game
from flip7.engine.player import Player


def hand_to_state(hand: list[int]) -> tuple[int, ...]:
    """Convert a hand to a 12-bit presence tuple: (has_1, has_2, ..., has_12)."""
    values = set(hand)
    return tuple(1 if v in values else 0 for v in range(1, 13))


class QAgent(Agent):
    """Tabular Q-learning agent using card presence as state."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9999,
        rng: random.Random | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng or random.Random()
        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(2)
        )

    def get_state(self, player: Player) -> tuple[int, ...]:
        return hand_to_state(player.hand)

    def choose_action(self, player: Player, opponent: Player, game: Game) -> Action:
        """Epsilon-greedy action selection."""
        state = self.get_state(player)
        if self.rng.random() < self.epsilon:
            return Action(self.rng.randint(0, 1))
        q_values = self.q_table[state]
        return Action(int(np.argmax(q_values)))

    def choose_action_greedy(self, player: Player, opponent: Player, game: Game) -> Action:
        """Greedy action selection (for evaluation)."""
        state = self.get_state(player)
        q_values = self.q_table[state]
        return Action(int(np.argmax(q_values)))

    def choose_action_from_state(self, state: tuple[int, ...]) -> int:
        """Epsilon-greedy selection from a raw state tuple. Returns 0=STAY, 1=DRAW."""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, 1)
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...] | None,
        done: bool,
    ) -> None:
        """Q-learning update rule."""
        if done or next_state is None:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            target - self.q_table[state][action]
        )

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    @classmethod
    def load(cls, path: str) -> "QAgent":
        agent = cls(epsilon=0.0)  # No exploration when loaded
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent.q_table = defaultdict(lambda: np.zeros(2), data)
        return agent
