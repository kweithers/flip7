from abc import ABC, abstractmethod

from flip7.engine.game import Action, Game
from flip7.engine.player import Player


class Agent(ABC):
    @abstractmethod
    def choose_action(self, player: Player, opponent: Player, game: Game) -> Action:
        """Decide whether to DRAW or STAY."""
        ...
