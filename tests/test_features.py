import random

import numpy as np
import pytest

from flip7.agents.random_agent import RandomAgent
from flip7.agents.threshold_agent import ThresholdAgent, draw_stats
from flip7.engine.features import OBS_DIMS, build_obs, draw_distribution, expected_draw_gain
from flip7.engine.game import Game
from flip7.environments.gym_env import Flip7Env


def _old_obs(player, opponent, game):
    """Verbatim copy of the pre-refactor 43-dim builder."""
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


def _old_gain(hand, game):
    """Pre-refactor ThresholdAgent arithmetic, written out independently."""
    from collections import Counter
    from flip7.engine.deck import build_full_deck
    counts = game.deck.card_counts()
    if sum(counts.values()) == 0:
        in_play = Counter(game._cards_in_play())
        full = Counter(build_full_deck())
        counts = {v: full[v] - in_play.get(v, 0) for v in range(1, 13)}
    held = set(hand)
    total = sum(counts.values())
    bust = sum(counts[v] for v in held)
    safe = total - bust
    if total == 0 or safe == 0:
        return 1.0, -sum(hand)
    e = sum(v * counts[v] for v in counts if v not in held) / safe
    p = bust / total
    return p, (1 - p) * e - p * sum(hand)


class Recorder:
    """Agent that records (player, opponent, game) states at decision points."""

    def __init__(self, inner):
        self.inner = inner
        self.states = []

    def choose_action(self, player, opponent, game):
        self.states.append(build_snapshot(player, opponent, game))
        return self.inner.choose_action(player, opponent, game)


def build_snapshot(player, opponent, game):
    return {v: build_obs(player, opponent, game, v) for v in OBS_DIMS} | {
        "old": _old_obs(player, opponent, game),
        "gain_me": _old_gain(player.hand, game),
        "gain_opp": _old_gain(opponent.hand, game),
        "deck_left": game.deck.remaining(),
    }


def _states(n_games=30):
    recs = [Recorder(RandomAgent(0.7)), Recorder(ThresholdAgent())]
    for s in range(n_games):
        Game(seed=s).play_game(recs)
    return recs[0].states + recs[1].states


def test_dims_and_ranges():
    for st in _states(10):
        for v, d in OBS_DIMS.items():
            assert st[v].shape == (d,) and st[v].dtype == np.float32
            if v == "base43":
                assert st[v].min() >= 0 and st[v].max() <= 1
            else:
                assert st[v].min() >= -1 and st[v].max() <= 1


def test_base43_identical_to_old_builder():
    for st in _states():
        assert st["base43"].tobytes() == st["old"].tobytes()
        assert st["added46"][:43].tobytes() == st["old"].tobytes()


def test_features_match_threshold_arithmetic():
    states = _states()
    assert any(st["deck_left"] == 0 for st in states), "need empty-deck states"
    for st in states:
        p, ev = st["gain_me"]
        assert st["added46"][43] == pytest.approx(p, abs=1e-6)
        assert st["added46"][44] == pytest.approx(np.clip(ev / 78, -1, 1), abs=1e-6)
        assert st["added46"][45] == pytest.approx(st["deck_left"] / 78, abs=1e-6)
        assert st["compact12"][0] == pytest.approx(p, abs=1e-6)
        assert st["compact12"][1] == pytest.approx(np.clip(ev / 78, -1, 1), abs=1e-6)
        po, evo = st["gain_opp"]
        assert st["compact12"][5] == pytest.approx(po, abs=1e-6)
        assert st["compact12"][6] == pytest.approx(np.clip(evo / 78, -1, 1), abs=1e-6)


def test_expected_draw_gain_matches_draw_stats():
    g = Game(seed=3)
    g.start_round()
    p, e = draw_stats(g.players[0].hand, draw_distribution(g))
    assert expected_draw_gain(g.players[0], g) == pytest.approx((1 - p) * e - p * g.players[0].hand_sum())


def _old_threshold_decision(agent, player, opponent, game):
    """Pre-refactor ThresholdAgent.choose_action, using the independent arithmetic."""
    mine = player.total_score + player.hand_sum()
    if opponent.is_done:
        opp_final = opponent.total_score + opponent.round_score()
        if mine >= 200 and mine > opp_final:
            return 0
        if opp_final >= 200 and opp_final > mine:
            return 1
    elif mine >= 200 and opponent.total_score + opponent.hand_sum() < mine:
        return 0
    return int(_old_gain(player.hand, game)[1] > agent.threshold(player, opponent))


def test_threshold_agent_decisions_unchanged():
    agent = ThresholdAgent()
    n = 0
    for s in range(40):
        class Check:
            def choose_action(self, player, opponent, game):
                nonlocal n
                a = agent.choose_action(player, opponent, game)
                assert int(a) == _old_threshold_decision(agent, player, opponent, game)
                n += 1
                return a

        Game(seed=s).play_game([Check(), RandomAgent(0.6)])
    assert n > 500


@pytest.mark.parametrize("variant", list(OBS_DIMS))
def test_env_and_agent_build_same_obs(variant):
    from flip7.agents.ppo_agent import PPOAgent
    env = Flip7Env(opponent_agent=RandomAgent(0.5), obs_variant=variant)
    agent = PPOAgent(model=None, obs_variant=variant)
    rng = random.Random(0)
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            g = env.game
            mine = agent._build_obs(g.players[0], g.players[1], g)
            assert mine.tobytes() == obs.tobytes()
            assert env.observation_space.shape == (OBS_DIMS[variant],)
            obs, _, term, trunc, _ = env.step(rng.random() < 0.6)
            done = term or trunc


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        Flip7Env(opponent_agent=RandomAgent(0.5), obs_variant="nope")
