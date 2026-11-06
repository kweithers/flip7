"""Fine-tune ppo_best with (curriculum) or without (control) end-game starts.

Usage:
    uv run python -m flip7.experiments.curriculum --arm curriculum --seed 0
    uv run python -m flip7.experiments.curriculum --arm control --seed 0 --steps 20000
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from flip7.agents.q_agent import QAgent
from flip7.environments.gym_env import Flip7Env
from flip7.experiments.common import build_frozen_ppo
from flip7.training.train_ppo import WinRateCallback, _load_obs_rms

MODELS = Path("models")
BASELINE = MODELS / "baseline"
FT = MODELS / "ft"
BASELINE_FILES = [
    "ppo_best.zip", "ppo_best_vecnorm.pkl", "ppo_final.zip",
    "vec_normalize_final.pkl", "q_table.pkl",
]
BURN_SIZES_PATH = Path("results/burn_deck_sizes.json")
SCENARIO_CONFIG_PATH = Path("results/scenario_config.json")

N_ENVS = 4
LEARNING_RATE = 3e-5
OPPONENT_WEIGHTS = {"ppo_best": 0.7, "q_agent": 0.3}


class EndgameSampler:
    """Samples an end-game start: scores (all <= 199), who acts first, burn."""

    TYPES = ("both_close", "trailing", "leading", "tied")

    def __init__(self, deck_sizes: list[int]):
        if not deck_sizes:
            raise ValueError("need a non-empty deck-size distribution")
        self.deck_sizes = np.asarray(deck_sizes)

    def __call__(self, rng: np.random.Generator) -> dict:
        kind = self.TYPES[int(rng.integers(len(self.TYPES)))]
        if kind == "both_close":
            me, opp = rng.integers(160, 200, size=2)
        elif kind == "trailing":
            me, opp = rng.integers(110, 171), rng.integers(175, 200)
        elif kind == "leading":
            me, opp = rng.integers(175, 200), rng.integers(110, 171)
        else:  # tied
            me = rng.integers(150, 200)
            opp = me + rng.integers(-5, 6)
        size = int(self.deck_sizes[int(rng.integers(len(self.deck_sizes)))])
        return {
            "type": kind,
            "my_score": int(min(me, 199)),
            "opp_score": int(min(opp, 199)),
            "first_parity": int(rng.integers(2)),
            "burn": min(78 - size, 77),
        }

    @classmethod
    def from_results(cls):
        return cls(json.loads(BURN_SIZES_PATH.read_text()))


def ensure_baseline():
    """Copy the original models aside once; training only ever reads from here."""
    BASELINE.mkdir(parents=True, exist_ok=True)
    for name in BASELINE_FILES:
        dst = BASELINE / name
        if not dst.exists():
            shutil.copy2(MODELS / name, dst)


def make_env_fns(scenario_prob: float, sampler, frozen, q_agent):
    pool = [(frozen, OPPONENT_WEIGHTS["ppo_best"]), (q_agent, OPPONENT_WEIGHTS["q_agent"])]

    def make():
        return Flip7Env(
            opponent_agent=frozen,
            opponent_pool=pool,
            scenario_prob=scenario_prob,
            scenario_sampler=sampler if scenario_prob > 0 else None,
        )

    return [make for _ in range(N_ENVS)]


def build_env(scenario_prob: float) -> VecNormalize:
    ensure_baseline()
    frozen = build_frozen_ppo(str(BASELINE / "ppo_best.zip"), str(BASELINE / "ppo_best_vecnorm.pkl"))
    q_agent = QAgent.load(str(BASELINE / "q_table.pkl"))
    sampler = EndgameSampler.from_results() if scenario_prob > 0 else None
    env = DummyVecEnv(make_env_fns(scenario_prob, sampler, frozen, q_agent))
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    # A fresh VecNormalize starts at mean 0 / var 1: load the frozen stats explicitly.
    env.obs_rms = _load_obs_rms(str(BASELINE / "ppo_best_vecnorm.pkl"))
    return env, q_agent


def load_model(env, seed: int) -> PPO:
    model = PPO.load(
        str(BASELINE / "ppo_best.zip"),
        env=env,
        device="cpu",
        custom_objects={"learning_rate": LEARNING_RATE},
        tensorboard_log="./tb_logs/",
    )
    # Seeds torch/numpy/random AND queues env seeds seed*1000 + i for the next reset.
    model.set_random_seed(seed * 1000)
    return model


class ScenarioShareCallback(BaseCallback):
    """Counts the share of env steps that belong to scenario-start episodes."""

    def __init__(self):
        super().__init__()
        self.scenario_steps = 0
        self.total_steps = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.total_steps += 1
            self.scenario_steps += bool(info.get("scenario"))
        return True

    @property
    def share(self) -> float:
        return self.scenario_steps / max(self.total_steps, 1)


def scenario_prob_for(arm: str) -> float:
    if arm == "control":
        return 0.0
    if arm == "curriculum":
        if not SCENARIO_CONFIG_PATH.exists():
            raise FileNotFoundError("run --calibrate-scenario first")
        return json.loads(SCENARIO_CONFIG_PATH.read_text())["scenario_prob"]
    raise ValueError(arm)


def finetune(arm: str, seed: int, steps: int = 500_000, scenario_prob: float | None = None,
             monitor: bool = True) -> Path:
    torch.set_num_threads(1)
    p = scenario_prob_for(arm) if scenario_prob is None else scenario_prob
    out = FT / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)

    env, q_agent = build_env(p)
    model = load_model(env, seed)

    share_cb = ScenarioShareCallback()
    callbacks = [
        share_cb,
        CheckpointCallback(save_freq=max(100_000 // N_ENVS, 1), save_path=str(out), name_prefix="ckpt"),
    ]
    if monitor:
        callbacks.append(WinRateCallback(q_agent, eval_freq=max(50_000 // N_ENVS, 1),
                                         n_eval_games=200, save_best=False, verbose=0))

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=CallbackList(callbacks), tb_log_name=f"ft_{arm}_s{seed}")
    model.save(str(out / "final"))

    meta = {
        "arm": arm, "seed": seed, "scenario_prob": p, "requested_steps": steps,
        "actual_steps": int(model.num_timesteps), "scenario_step_share": share_cb.share,
        "seconds": time.time() - t0, "learning_rate": LEARNING_RATE,
        "ent_coef": float(model.ent_coef), "opponent_weights": OPPONENT_WEIGHTS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return out


def measure_scenario_share(scenario_prob: float, n_steps: int = 40_000, seed: int = 0) -> float:
    """Rollout-only (no learning) share of steps from scenario episodes, using ppo_best."""
    env, _ = build_env(scenario_prob)
    model = load_model(env, seed)
    obs = env.reset()
    scen = total = 0
    for _ in range(n_steps // N_ENVS):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, _, infos = env.step(action)
        for info in infos:
            total += 1
            scen += bool(info.get("scenario"))
    return scen / total


def calibrate_scenario(target: float = 0.30):
    """Bisect scenario_prob so ~target of training steps come from scenario starts."""
    lo, hi = 0.0, 1.0
    tried = []
    for _ in range(7):
        mid = (lo + hi) / 2
        share = measure_scenario_share(mid)
        tried.append((round(mid, 4), round(share, 4)))
        print(f"scenario_prob={mid:.4f} -> step share {share:.3f}")
        if share < target:
            lo = mid
        else:
            hi = mid
    best = min(tried, key=lambda t: abs(t[1] - target))
    cfg = {"target_step_share": target, "scenario_prob": best[0], "measured_share": best[1], "tried": tried}
    SCENARIO_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "curriculum"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--calibrate-scenario", action="store_true")
    ap.add_argument("--no-monitor", action="store_true")
    args = ap.parse_args()
    if args.calibrate_scenario:
        calibrate_scenario()
    elif args.arm:
        finetune(args.arm, args.seed, args.steps, monitor=not args.no_monitor)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
