"""Rule-based, score-aware agent used as a held-out evaluation opponent.

Base rule: draw while the one-step expected change in round score is above a
threshold. The threshold is 0 normally and goes negative (more risk) when
trailing an opponent who is near 200. Game-ending overrides come first.
"""

from flip7.agents.base import Agent
from flip7.engine.features import (  # noqa: F401  (re-exported)
    draw_distribution,
    draw_stats,
    expected_draw_gain,
)
from flip7.engine.game import Action, Game
from flip7.engine.player import Player

WIN = Game.WINNING_SCORE
NEAR_WIN = 150


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
