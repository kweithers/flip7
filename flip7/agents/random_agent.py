import random

from flip7.agents.base import Agent
from flip7.engine.game import Action, Game
from flip7.engine.player import Player


class RandomAgent(Agent):
    """Draws with a fixed probability, otherwise stays."""

    def __init__(self, draw_prob: float = 0.5, rng: random.Random | None = None):
        self.draw_prob = draw_prob
        self.rng = rng or random.Random()

    def choose_action(self, player: Player, opponent: Player, game: Game) -> Action:
        if self.rng.random() < self.draw_prob:
            return Action.DRAW
        return Action.STAY
