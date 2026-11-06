import numpy as np
from gymnasium.utils import seeding

from flip7.agents.base import Agent
from flip7.agents.threshold_agent import ThresholdAgent
from flip7.engine.game import Action, Game
from flip7.environments.gym_env import Flip7Env
from flip7.experiments.curriculum import EndgameSampler


class Recorder(Agent):
    def __init__(self):
        self.seen = []

    def choose_action(self, player, opponent, game):
        self.seen.append((player.total_score, opponent.total_score))
        return Action.STAY


def fixed(sc):
    return lambda rng: dict(sc)


def test_default_env_games_unchanged():
    # The default env must draw exactly one game seed per reset, as before.
    env = Flip7Env(opponent_agent=ThresholdAgent())
    rng, _ = seeding.np_random(5)
    for i in range(5):
        env.reset(seed=5 if i == 0 else None)
        expected = Game(seed=int(rng.integers(0, 2**31))).deck._cards
        remaining = env.game.deck._cards
        assert expected[: len(remaining)] == remaining
        assert not env.is_scenario


def test_scenario_sets_scores_and_first_player_agent():
    sc = {"my_score": 180, "opp_score": 170, "first_parity": 0, "burn": 30}
    env = Flip7Env(opponent_agent=Recorder(), scenario_prob=1.0, scenario_sampler=fixed(sc))
    env.reset(seed=0)
    me, opp = env.game.players
    assert (me.total_score, opp.total_score) == (180, 170)
    assert env.game.current_player_idx == 0
    assert env.game.round_number == 1
    assert env.opponent_agent.seen == []
    # Card accounting: deck + hands = 78 - burned.
    assert env.game.deck.remaining() + len(me.hand) + len(opp.hand) == 78 - 30
    assert env.is_scenario


def test_scenario_opponent_first_sees_scenario_scores():
    rec = Recorder()
    sc = {"my_score": 150, "opp_score": 195, "first_parity": 1, "burn": 10}
    env = Flip7Env(opponent_agent=rec, scenario_prob=1.0, scenario_sampler=fixed(sc))
    env.reset(seed=1)
    assert rec.seen == [(195, 150)]
    assert env.game.current_player_idx == 0  # back to the agent


def test_burn_77_reshuffles_cleanly():
    sc = {"my_score": 160, "opp_score": 160, "first_parity": 0, "burn": 77}
    env = Flip7Env(opponent_agent=Recorder(), scenario_prob=1.0, scenario_sampler=fixed(sc))
    env.reset(seed=2)
    obs, *_ = env.step(1)  # draw: forces a reshuffle
    assert obs.shape == (43,)


def test_sampler_scores_valid():
    s = EndgameSampler([0, 10, 23, 40])
    rng = np.random.default_rng(0)
    for _ in range(2000):
        sc = s(rng)
        assert 0 <= sc["my_score"] <= 199 and 0 <= sc["opp_score"] <= 199
        assert sc["first_parity"] in (0, 1)
        assert 0 <= sc["burn"] <= 77


def test_opponent_pool_sampling():
    a, b = Recorder(), Recorder()
    env = Flip7Env(opponent_agent=a, opponent_pool=[(a, 0.7), (b, 0.3)])
    picks = []
    for i in range(300):
        env.reset(seed=i)
        picks.append(env.opponent_agent is a)
    assert 0.6 < np.mean(picks) < 0.8
