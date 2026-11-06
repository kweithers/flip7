from flip7.agents.threshold_agent import (
    ThresholdAgent,
    draw_distribution,
    draw_stats,
)
from flip7.engine.game import Action, Game


def setup(my_total, my_hand, opp_total, opp_hand, opp_stayed=False, opp_busted=False):
    game = Game(seed=0)
    me, opp = game.players
    me.total_score, me.hand = my_total, list(my_hand)
    opp.total_score, opp.hand = opp_total, list(opp_hand)
    opp.stayed, opp.busted = opp_stayed, opp_busted
    return game, me, opp


def test_empty_deck_uses_reshuffle_distribution():
    game, me, opp = setup(0, [12], 0, [11])
    game.deck._cards = []
    counts = draw_distribution(game)
    assert counts[12] == 11 and counts[11] == 10 and counts[1] == 1
    p, e = draw_stats(me.hand, counts)
    assert 0 < p < 1 and e > 0


def test_override_order_staying_wins():
    # Lead 190-100, opponent stayed on 25, I hold 15 -> staying wins (205 vs 125).
    game, me, opp = setup(190, [7, 8], 100, [12, 13], opp_stayed=True)
    assert ThresholdAgent().choose_action(me, opp, game) == Action.STAY


def test_must_draw_when_staying_loses():
    # Opponent stayed and will finish on 205; I would finish on 190.
    game, me, opp = setup(180, [4, 6], 190, [15], opp_stayed=True)
    assert ThresholdAgent().choose_action(me, opp, game) == Action.DRAW


def test_both_over_200_higher_score_needed():
    # Opponent stayed and finishes on 210; staying gives me 205 -> must draw.
    game, me, opp = setup(195, [10], 195, [15], opp_stayed=True)
    assert ThresholdAgent().choose_action(me, opp, game) == Action.DRAW


def test_stays_on_high_risk_hand_early_game():
    game, me, opp = setup(0, [12, 11, 10, 9, 8, 7], 0, [1])
    assert ThresholdAgent().choose_action(me, opp, game) == Action.STAY


def test_draws_on_low_risk_hand():
    game, me, opp = setup(0, [1], 0, [2])
    assert ThresholdAgent().choose_action(me, opp, game) == Action.DRAW


def test_aggression_when_trailing_near_win():
    hand = [12, 11, 10, 3]
    game, me, opp = setup(120, hand, 180, [5])
    calm = ThresholdAgent(aggression=0.0).choose_action(me, opp, game)
    bold = ThresholdAgent(aggression=1.0).choose_action(me, opp, game)
    assert calm == Action.STAY and bold == Action.DRAW
