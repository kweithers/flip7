"""Rule-based, score-aware agent used as a held-out evaluation opponent.

Base rule: draw while the one-step expected change in round score is above a
threshold. The threshold is 0 normally and goes negative (more risk) when
trailing an opponent who is near 200. Game-ending overrides come first.
"""

from collections import Counter

from flip7.agents.base import Agent
from flip7.engine.deck import build_full_deck
from flip7.engine.game import Action, Game
from flip7.engine.player import Player

WIN = Game.WINNING_SCORE
NEAR_WIN = 150


def draw_distribution(game: Game) -> dict[int, int]:
    """Counts of each value the next draw can come from.

    If the deck is empty, the next draw triggers a reshuffle of the full deck
    minus the cards currently in play (mirrors ``Deck.reshuffle``).
    """
    counts = game.deck.card_counts()
    if sum(counts.values()) > 0:
        return counts
    in_play = Counter(game._cards_in_play())
    full = Counter(build_full_deck())
    return {v: full[v] - in_play.get(v, 0) for v in range(1, 13)}


def draw_stats(hand: list[int], counts: dict[int, int]) -> tuple[float, float]:
    """Return (p_bust, expected value of the drawn card given it is safe)."""
    held = set(hand)
    total = sum(counts.values())
    if total == 0:
        return 1.0, 0.0
    bust = sum(counts[v] for v in held)
    safe = total - bust
    if safe == 0:
        return 1.0, 0.0
    e_safe = sum(v * counts[v] for v in counts if v not in held) / safe
    return bust / total, e_safe


def expected_draw_gain(player: Player, game: Game) -> float:
    """(1 - p) * E[card | safe] - p * hand_sum."""
    p, e_safe = draw_stats(player.hand, draw_distribution(game))
    return (1 - p) * e_safe - p * player.hand_sum()


class ThresholdAgent(Agent):
    def __init__(self, aggression: float = 0.2, gap_cap: int = 60):
        self.aggression = aggression
        self.gap_cap = gap_cap

    def threshold(self, player: Player, opponent: Player) -> float:
        gap = opponent.total_score - player.total_score
        if gap > 0 and opponent.total_score >= NEAR_WIN:
            return -self.aggression * min(gap, self.gap_cap)
        return 0.0

    def choose_action(self, player: Player, opponent: Player, game: Game) -> Action:
        mine_if_stay = player.total_score + player.hand_sum()

        if opponent.is_done:
            opp_final = opponent.total_score + opponent.round_score()
            # 1. Staying wins the game outright.
            if mine_if_stay >= WIN and mine_if_stay > opp_final:
                return Action.STAY
            # 2. Staying loses the game: must draw.
            if opp_final >= WIN and opp_final > mine_if_stay:
                return Action.DRAW
        else:
            # 3. Staying reaches 200 and the opponent is currently behind.
            opp_now = opponent.total_score + opponent.hand_sum()
            if mine_if_stay >= WIN and opp_now < mine_if_stay:
                return Action.STAY

        # 4. Base rule.
        if expected_draw_gain(player, game) > self.threshold(player, opponent):
            return Action.DRAW
        return Action.STAY
