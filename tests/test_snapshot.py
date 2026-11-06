from flip7.agents.random_agent import RandomAgent
from flip7.agents.threshold_agent import ThresholdAgent
from flip7.engine.deck import Deck
from flip7.engine.game import Game
from flip7.engine.snapshot import capture, restore
import random

import pytest


def play_rounds(game, agents, n):
    for _ in range(n):
        if game.game_over:
            break
        game.play_round(agents)
        game.current_player_idx = game.round_number % 2


def test_burn_removes_cards_without_reshuffle():
    d = Deck(random.Random(0))
    d.burn(10)
    assert d.remaining() == 68
    with pytest.raises(ValueError):
        d.burn(69)


def test_restore_reproduces_rest_of_game():
    agents = [ThresholdAgent(), ThresholdAgent(aggression=0.5)]
    game = Game(seed=7)
    play_rounds(game, agents, 4)
    snap = capture(game)

    a = restore(snap)
    b = restore(snap)
    ra = a.play_game(agents)
    rb = b.play_game(agents)
    assert ra.winner == rb.winner and ra.scores == rb.scores

    # Continuing the original game gives the same result too.
    ro = game.play_game(agents)
    assert ro.scores == ra.scores


def test_restore_is_independent_of_original():
    game = Game(seed=3)
    play_rounds(game, [ThresholdAgent(), ThresholdAgent()], 2)
    snap = capture(game)
    g = restore(snap)
    g.deck.draw()
    assert len(snap["deck_cards"]) == g.deck.remaining() + 1


def test_restore_through_reshuffle():
    # Play long enough that the deck reshuffles after the snapshot.
    agents = [RandomAgent(0.7, random.Random(1)), RandomAgent(0.7, random.Random(2))]
    game = Game(seed=11)
    play_rounds(game, agents, 3)
    snap = capture(game)
    det = [ThresholdAgent(), ThresholdAgent(aggression=1.0)]
    assert restore(snap).play_game(det).scores == restore(snap).play_game(det).scores
