import os

import numpy as np
import pytest

from flip7.engine.game import Game
from flip7.experiments.common import build_frozen_ppo, is_close_endgame

MODEL = "models/ppo_best.zip"
STATS = "models/ppo_best_vecnorm.pkl"


def test_is_close_endgame():
    assert is_close_endgame([150, 190])
    assert not is_close_endgame([150, 191])
    assert not is_close_endgame([200, 190])
    assert not is_close_endgame([149, 160])


@pytest.mark.skipif(not os.path.exists(MODEL), reason="model not present")
def test_fast_agent_matches_predict():
    fast = build_frozen_ppo(MODEL, STATS, fast=True)
    slow = build_frozen_ppo(MODEL, STATS, fast=False)
    rng = np.random.default_rng(0)
    for seed in range(30):
        game = Game(seed=seed)
        game.start_round()
        me, opp = game.players
        me.total_score, opp.total_score = (int(x) for x in rng.integers(0, 199, 2))
        for _ in range(4):
            assert fast.choose_action(me, opp, game) == slow.choose_action(me, opp, game)
            if game.step(fast.choose_action(me, opp, game)):
                break
