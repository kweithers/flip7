"""PPO training pipeline for Flip 7.

Stage 1: Train against Q-learning opponent (1M steps)
Stage 2: Self-play fine-tuning (500K steps)
"""

import json
import pickle
import random

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from flip7.agents.ppo_agent import PPOAgent
from flip7.agents.q_agent import QAgent
from flip7.agents.random_agent import RandomAgent
from flip7.environments.gym_env import Flip7Env
from flip7.engine.game import Game


def make_env(opponent_agent, seed):
    def _init():
        env = Flip7Env(opponent_agent=opponent_agent)
        env.reset(seed=seed)
        return env
    return _init


class WinRateCallback(BaseCallback):
    """Log win rate using a separate eval environment."""

    def __init__(self, opponent_agent, eval_freq=50_000, n_eval_games=200, verbose=1,
                 save_best=True, best_model_path="models/ppo_best"):
        super().__init__(verbose)
        self.opponent_agent = opponent_agent
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.save_best = save_best
        self.best_model_path = best_model_path
        self.best_win_rate = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            eval_env = Flip7Env(opponent_agent=self.opponent_agent)
            vec_env = self.training_env

            wins = 0
            for i in range(self.n_eval_games):
                obs, _ = eval_env.reset(seed=i + 10000)
                # Normalize using training stats
                obs = vec_env.normalize_obs(np.array([obs]))[0]

                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = eval_env.step(int(action))
                    obs = vec_env.normalize_obs(np.array([obs]))[0]
                    done = terminated or truncated

                if eval_env.game.result and eval_env.game.result.winner == 0:
                    wins += 1

            win_rate = wins / self.n_eval_games
            self.logger.record("eval/win_rate", win_rate)
            if self.verbose:
                print(f"Step {self.n_calls}: Win rate = {win_rate:.3f}")

            if self.save_best and win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.model.save(self.best_model_path)
                vecnorm_path = f"{self.best_model_path}_vecnorm.pkl"
                vec_env = self.training_env
                with open(vecnorm_path, "wb") as f:
                    pickle.dump({
                        "obs_rms": vec_env.obs_rms,
                        "ret_rms": vec_env.ret_rms,
                        "clip_obs": vec_env.clip_obs,
                        "norm_obs": vec_env.norm_obs,
                    }, f)
                if self.verbose:
                    print(f"  New best model saved (win rate: {win_rate:.3f})")
        return True


def train_stage1(
    q_agent_path: str = "models/q_table.pkl",
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    seed: int = 42,
) -> tuple[PPO, VecNormalize]:
    """Stage 1: Train PPO against Q-learning opponent."""
    print("=" * 60)
    print("Stage 1: Training PPO vs Q-learning agent")
    print("=" * 60)

    q_agent = QAgent.load(q_agent_path)

    env = DummyVecEnv([make_env(q_agent, seed + i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
        verbose=1,
        seed=seed,
        tensorboard_log="./tb_logs/",
    )

    callback = WinRateCallback(q_agent, eval_freq=10_000, n_eval_games=200)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("models/ppo_stage1")
    env.save("models/vec_normalize_stage1.pkl")

    return model, env


def train_stage2(
    model: PPO,
    vec_env: VecNormalize,
    total_timesteps: int = 500_000,
    update_opponent_every: int = 100_000,
    seed: int = 42,
) -> tuple[PPO, VecNormalize]:
    """Stage 2: Self-play fine-tuning."""
    print("=" * 60)
    print("Stage 2: Self-play fine-tuning")
    print("=" * 60)

    frozen_model = PPO.load("models/ppo_stage1")
    frozen_agent = PPOAgent(frozen_model)

    n_envs = 4
    env = DummyVecEnv([make_env(frozen_agent, seed + i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # Copy normalization stats from stage 1
    env.obs_rms = vec_env.obs_rms
    env.ret_rms = vec_env.ret_rms

    model.set_env(env)

    class SelfPlayCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.steps_since_update = 0

        def _on_step(self) -> bool:
            self.steps_since_update += 1
            if self.steps_since_update >= update_opponent_every:
                print(f"  Updating frozen opponent at step {self.n_calls}")
                self.model.save("models/ppo_selfplay_temp")
                new_frozen = PPO.load("models/ppo_selfplay_temp")
                frozen_agent.model = new_frozen
                self.steps_since_update = 0
            return True

    callback = SelfPlayCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("models/ppo_final")
    env.save("models/vec_normalize_final.pkl")

    return model, env


def _run_eval(agent1, agent2, num_games):
    """Run head-to-head evaluation."""
    rng = random.Random(999)
    wins = 0
    total_score = 0
    total_rounds = 0

    for i in range(num_games):
        game = Game(seed=rng.randint(0, 2**31))
        if i % 2 == 0:
            agents = [agent1, agent2]
            result = game.play_game(agents)
            if result.winner == 0:
                wins += 1
            total_score += result.scores[0]
        else:
            agents = [agent2, agent1]
            result = game.play_game(agents)
            if result.winner == 1:
                wins += 1
            total_score += result.scores[1]
        total_rounds += game.round_number

    win_rate = wins / num_games
    avg_score = total_score / num_games
    avg_rounds = total_rounds / num_games
    print(f"  Win rate: {win_rate:.3f}")
    print(f"  Avg score: {avg_score:.1f}")
    print(f"  Avg game length: {avg_rounds:.1f} rounds")


def _load_obs_rms(vec_normalize_path: str):
    """Load obs_rms from either a VecNormalize pickle or our custom dict format."""
    with open(vec_normalize_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data["obs_rms"]
    # Standard VecNormalize object
    return data.obs_rms


def evaluate_model(model_path: str, vec_normalize_path: str, num_games: int = 1000):
    """Evaluate the trained PPO model."""
    model = PPO.load(model_path)

    obs_rms = _load_obs_rms(vec_normalize_path)
    obs_rms_dict = {
        "mean": obs_rms.mean.astype(np.float32),
        "var": obs_rms.var.astype(np.float32),
    }
    ppo_agent = PPOAgent(model, obs_rms_dict)

    q_agent = QAgent.load("models/q_table.pkl")
    print("\nPPO vs Q-agent:")
    _run_eval(ppo_agent, q_agent, num_games)

    random_agent = RandomAgent(draw_prob=0.5, rng=random.Random(789))
    print("\nPPO vs Random (0.5):")
    _run_eval(ppo_agent, random_agent, num_games)


def export_normalize_stats(vec_normalize_path: str, output_path: str):
    """Export VecNormalize stats as JSON for the webapp."""
    obs_rms = _load_obs_rms(vec_normalize_path)
    stats = {
        "mean": obs_rms.mean.tolist(),
        "var": obs_rms.var.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(stats, f)
    print(f"Saved normalization stats to {output_path}")


if __name__ == "__main__":
    model, vec_env = train_stage1(total_timesteps=5_000_000)
    model, vec_env = train_stage2(model, vec_env, total_timesteps=2_000_000)
    evaluate_model("models/ppo_final.zip", "models/vec_normalize_final.pkl")
    export_normalize_stats("models/vec_normalize_final.pkl", "models/normalize_stats.json")
