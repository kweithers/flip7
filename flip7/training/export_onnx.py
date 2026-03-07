"""Export trained PPO model to ONNX format for browser inference."""

import json

import numpy as np
import torch
import torch.onnx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from flip7.agents.random_agent import RandomAgent
from flip7.environments.gym_env import Flip7Env
from flip7.training.train_ppo import make_env


class PolicyWrapper(torch.nn.Module):
    """Wraps the SB3 policy for clean ONNX export (obs -> action logits)."""

    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs):
        features = self.mlp_extractor.forward_actor(obs)
        return self.action_net(features)


def export_onnx(
    model_path: str = "models/ppo_final.zip",
    vec_normalize_path: str = "models/vec_normalize_final.pkl",
    onnx_output: str = "models/ppo_flip7.onnx",
    stats_output: str = "models/normalize_stats.json",
):
    # Load model
    model = PPO.load(model_path)
    policy = model.policy

    # Wrap for export
    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, 43)
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_output,
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to {onnx_output}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(onnx_output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")

    # Export normalization stats
    dummy_env = DummyVecEnv([make_env(RandomAgent(), 0)])
    vec_env = VecNormalize.load(vec_normalize_path, dummy_env)
    stats = {
        "mean": vec_env.obs_rms.mean.tolist(),
        "var": vec_env.obs_rms.var.tolist(),
    }
    with open(stats_output, "w") as f:
        json.dump(stats, f)
    print(f"Exported normalization stats to {stats_output}")

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        test_obs = torch.randn(1, 43)
        with torch.no_grad():
            pytorch_out = wrapper(test_obs).numpy()
        session = ort.InferenceSession(onnx_output)
        onnx_out = session.run(None, {"observation": test_obs.numpy()})[0]
        diff = np.abs(pytorch_out - onnx_out).max()
        print(f"Max difference between PyTorch and ONNX outputs: {diff:.6e}")
        assert diff < 1e-5, f"ONNX output differs too much: {diff}"
        print("Verification passed!")
    except ImportError:
        print("onnxruntime not available, skipping runtime verification")


if __name__ == "__main__":
    export_onnx()
