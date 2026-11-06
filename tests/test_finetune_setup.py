import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/ppo_best.zip"), reason="models not present"
)


@pytest.fixture(scope="module")
def setup():
    from flip7.experiments.curriculum import build_env, load_model
    env, _ = build_env(0.0)
    model = load_model(env, seed=0)
    return env, model


def test_learning_rate_override(setup):
    _, model = setup
    assert model.lr_schedule(1.0) == pytest.approx(3e-5)
    assert model.ent_coef == pytest.approx(0.01)


def test_normalization_matches_ppo_agent(setup):
    from flip7.experiments.common import build_frozen_ppo
    env, _ = setup
    agent = build_frozen_ppo("models/baseline/ppo_best.zip", "models/baseline/ppo_best_vecnorm.pkl")
    x = np.random.default_rng(0).random((5, 43)).astype(np.float32)
    np.testing.assert_allclose(env.normalize_obs(x), agent._normalize_obs(x), rtol=1e-5, atol=1e-5)


def test_obs_rms_frozen_and_seeding_reproducible():
    from flip7.experiments.curriculum import build_env, load_model
    buffers = []
    for seed in (0, 0, 1):
        env, _ = build_env(0.0)
        before = env.obs_rms.mean.copy()
        model = load_model(env, seed=seed)
        model.learn(total_timesteps=1)  # one rollout of 4 x 2048 + one update
        np.testing.assert_array_equal(env.obs_rms.mean, before)
        buffers.append(model.rollout_buffer.actions.copy())
    np.testing.assert_array_equal(buffers[0], buffers[1])
    assert not np.array_equal(buffers[0], buffers[2])


def test_baseline_models_untouched():
    import hashlib
    from flip7.experiments.curriculum import ensure_baseline
    def h(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    before = {p: h(p) for p in ("models/ppo_best.zip", "models/ppo_final.zip")}
    ensure_baseline()
    assert {p: h(p) for p in before} == before
    assert h("models/baseline/ppo_best.zip") == before["models/ppo_best.zip"]
