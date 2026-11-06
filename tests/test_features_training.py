import os
import pickle
from pathlib import Path

import numpy as np

from flip7.experiments import features as F


def test_checkpoints_have_own_stats_and_stay_in_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "OUT", tmp_path)
    watched = [p for p in Path("models").glob("ppo_*") if p.is_file()]
    before = {p: os.stat(p).st_mtime_ns for p in watched}

    out = F.train("C", seed=7, steps=8000, ckpt_every=4096)

    assert out.parent == tmp_path
    names = sorted(p.name for p in out.iterdir())
    assert names == ["ckpt_4096.zip", "ckpt_4096_vecnorm.pkl", "final.zip",
                     "final_vecnorm.pkl", "meta.json"]
    ck = pickle.load(open(out / "ckpt_4096_vecnorm.pkl", "rb"))["obs_rms"]
    fin = pickle.load(open(out / "final_vecnorm.pkl", "rb"))["obs_rms"]
    assert ck.mean.shape == fin.mean.shape == (12,)
    assert ck.count < fin.count  # stats kept updating after the checkpoint
    assert not np.allclose(ck.mean, fin.mean)
    assert F.checkpoints("C", 7) == [(4096, "ckpt_4096"), (8192, "final")]
    agent = F.load_candidate("C", 7, "ckpt_4096")
    assert agent.obs_variant == "compact12"
    assert np.allclose(agent.obs_rms["mean"], ck.mean)
    assert {p: os.stat(p).st_mtime_ns for p in watched} == before
