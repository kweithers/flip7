import random
from enum import IntEnum

from flip7.engine.deck import Deck
from flip7.engine.player import Player


class Action(IntEnum):
    STAY = 0
    DRAW = 1


class GameResult:
    def __init__(self, winner: int, scores: list[int]):
        self.winner = winner  # 0 or 1
        self.scores = scores


class Game:
    """Two-player Flip 7 game engine.

    Players alternate turns within each round. Each turn a player
    either DRAWs (risking a bust) or STAYs (banking their hand sum).
    First to 200 wins; if both cross 200 in the same round, highest score wins.
    """

    WINNING_SCORE = 200

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.deck = Deck(self.rng)
        self.players = [Player(), Player()]
        self.current_player_idx = 0
        self.round_number = 0
        self.game_over = False
        self.result: GameResult | None = None

    def _cards_in_play(self) -> list[int]:
        """All cards currently in players' hands."""
        cards = []
        for p in self.players:
            cards.extend(p.hand)
        return cards

    def start_round(self) -> None:
        """Begin a new round: reset hands and deal one card to each player."""
        self.round_number += 1
        for p in self.players:
            p.reset_round()
        # Deal one card to each player
        for p in self.players:
            card = self.deck.draw(self._cards_in_play())
            p.add_card(card)

    def get_current_player(self) -> Player:
        return self.players[self.current_player_idx]

    def get_opponent(self, player_idx: int = None) -> Player:
        if player_idx is None:
            player_idx = self.current_player_idx
        return self.players[1 - player_idx]

    def step(self, action: Action) -> bool:
        """Execute one action for the current player.

        Returns True if the round is over after this action.
        """
        player = self.get_current_player()

        if action == Action.DRAW:
            card = self.deck.draw(self._cards_in_play())
            player.add_card(card)
        else:
            player.stay()

        # Check if round is over (both players done)
        if all(p.is_done for p in self.players):
            self._end_round()
            return True

        # Advance to next player who isn't done
        self._advance_turn()
        return False

    def _advance_turn(self) -> None:
        """Move to the next player who hasn't stayed or busted."""
        self.current_player_idx = 1 - self.current_player_idx
        # If the other player is done, switch back
        if self.players[self.current_player_idx].is_done:
            self.current_player_idx = 1 - self.current_player_idx

    def _end_round(self) -> None:
        """Tally round scores and check for game end."""
        for p in self.players:
            p.total_score += p.round_score()

        # Check if game is over
        p0_over = self.players[0].total_score >= self.WINNING_SCORE
        p1_over = self.players[1].total_score >= self.WINNING_SCORE

        if p0_over or p1_over:
            if p0_over and p1_over:
                # Both over 200: tied scores mean play another round
                if self.players[0].total_score == self.players[1].total_score:
                    return
                winner = 0 if self.players[0].total_score > self.players[1].total_score else 1
            elif p0_over:
                winner = 0
            else:
                winner = 1
            self.game_over = True
            self.result = GameResult(
                winner=winner,
                scores=[p.total_score for p in self.players],
            )

    def is_round_over(self) -> bool:
        return all(p.is_done for p in self.players)

    def play_round(self, agents: list) -> None:
        """Play a full round with two agents that implement choose_action().

        Each agent's choose_action receives (player, opponent, game) and returns Action.
        """
        self.start_round()
        while not self.is_round_over():
            player = self.get_current_player()
            opponent = self.get_opponent()
            action = agents[self.current_player_idx].choose_action(
                player, opponent, self
            )
            self.step(action)

    def play_game(self, agents: list) -> GameResult:
        """Play a full game until someone wins. Returns GameResult."""
        while not self.game_over:
            self.play_round(agents)
            # Alternate who goes first each round
            self.current_player_idx = self.round_number % 2
        return self.result
