"""Observation builders shared by the gym env and the PPO agent wrappers.

Variants:
  base43    — the original 43-dim observation (unchanged).
  added46   — base43 + my P(bust), my E(draw)/78 (clipped to [-1, 1]), cards left/78.
  compact12 — me: P(bust), E(draw)/78, hand sum/78, hand size/12, total/200;
              opponent: P(bust), E(draw)/78, hand sum/78, total/200, stayed;
              shared: cards left/78, (my total - opp total)/200.

P(bust) and E(draw) are the exact quantities ThresholdAgent uses:
E(draw) = (1 - p) * E[card | safe] - p * hand_sum, with an empty deck treated
as the post-reshuffle distribution.
"""

from collections import Counter

import numpy as np

from flip7.engine.deck import build_full_deck

OBS_DIMS = {"base43": 43, "added46": 46, "compact12": 12}
VARIANTS = tuple(OBS_DIMS)
_FULL_COUNTS = Counter(build_full_deck())


def draw_distribution(game) -> dict[int, int]:
    """Counts of each value the next draw can come from.

    If the deck is empty, the next draw triggers a reshuffle of the full deck
    minus the cards currently in play (mirrors ``Deck.reshuffle``).
    """
    counts = game.deck.card_counts()
    if sum(counts.values()) > 0:
        return counts
    in_play = Counter(game._cards_in_play())
    return {v: _FULL_COUNTS[v] - in_play.get(v, 0) for v in range(1, 13)}


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


def draw_gain(hand: list[int], counts: dict[int, int]) -> tuple[float, float]:
    """Return (p_bust, E(draw)) for a hand against a draw distribution."""
    p, e_safe = draw_stats(hand, counts)
    return p, (1 - p) * e_safe - p * sum(hand)


def expected_draw_gain(player, game) -> float:
    """(1 - p) * E[card | safe] - p * hand_sum."""
    return draw_gain(player.hand, draw_distribution(game))[1]


def _base43(player, opponent, game) -> np.ndarray:
    deck_counts = game.deck.card_counts()
    obs = np.zeros(43, dtype=np.float32)
    for card in player.hand:
        obs[card - 1] = 1.0
    obs[12] = len(player.hand) / 12.0
    obs[13] = sum(player.hand) / 78.0
    obs[14] = min(player.total_score / 200.0, 1.0)
    for card in opponent.hand:
        obs[15 + card - 1] = 1.0
    obs[27] = len(opponent.hand) / 12.0
    obs[28] = sum(opponent.hand) / 78.0
    obs[29] = min(opponent.total_score / 200.0, 1.0)
    obs[30] = 1.0 if opponent.stayed else 0.0
    for v in range(1, 13):
        obs[31 + v - 1] = deck_counts.get(v, 0) / v
    return obs


def _ev(x: float) -> float:
    return float(np.clip(x / 78.0, -1.0, 1.0))


def build_obs(player, opponent, game, variant: str = "base43") -> np.ndarray:
    if variant == "base43":
        return _base43(player, opponent, game)
    counts = draw_distribution(game)
    cards_left = game.deck.remaining() / 78.0
    p_me, ev_me = draw_gain(player.hand, counts)
    if variant == "added46":
        obs = np.empty(46, dtype=np.float32)
        obs[:43] = _base43(player, opponent, game)
        obs[43] = p_me
        obs[44] = _ev(ev_me)
        obs[45] = cards_left
        return obs
    if variant == "compact12":
        p_opp, ev_opp = draw_gain(opponent.hand, counts)
        gap = (player.total_score - opponent.total_score) / 200.0
        return np.array([
            p_me, _ev(ev_me), sum(player.hand) / 78.0, len(player.hand) / 12.0,
            min(player.total_score / 200.0, 1.0),
            p_opp, _ev(ev_opp), sum(opponent.hand) / 78.0,
            min(opponent.total_score / 200.0, 1.0), 1.0 if opponent.stayed else 0.0,
            cards_left, float(np.clip(gap, -1.0, 1.0)),
        ], dtype=np.float32)
    raise ValueError(f"unknown obs variant {variant!r}; expected one of {VARIANTS}")


def obs_bounds(variant: str) -> tuple[float, float]:
    """Box bounds: base43 keeps its original [0, 1]; others contain [-1, 1] values."""
    return (0.0, 1.0) if variant == "base43" else (-1.0, 1.0)
