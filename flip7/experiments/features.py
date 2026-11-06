"""P(bust)/E(draw) observation-feature ablation (docs/pbust_edraw_features_plan.md).

Arms (all trained from scratch with the Stage 1 recipe vs the Q-agent):
  A = base43, B = added46, C = compact12.

Usage:
    uv run python -m flip7.experiments.features --train A --seed 0 [--steps 20000]
    uv run python -m flip7.experiments.features --all           # train + validate + test, 4 procs
    uv run python -m flip7.experiments.features --report
"""

import argparse
import json
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from flip7.agents.base import Agent
from flip7.engine.features import expected_draw_gain

ARMS = {"A": "base43", "B": "added46", "C": "compact12"}
SEEDS = [0, 1, 2, 3, 4]
STEPS = 2_000_000
CKPT_EVERY = 250_000
N_ENVS = 4
OUT = Path("models/feat")
RESULTS = Path("results")

VAL_SEEDS = range(500_000, 500_500)       # 500 x 2, checkpoint selection only
RR_SEEDS = range(600_000, 601_000)        # round-robin, 1,000 x 2
# Test seeds are reused from the end-game experiment (disjoint from the above):
#   P1 300000-302499 (2,500 x 2), checks 400000-400999 (1,000 x 2).

EV_CLEAR = 3.0
DECK_BUCKETS = [("<20", 0, 20), ("20-40", 20, 41), ("40+", 41, 79)]


def run_dir(arm, seed) -> Path:
    return OUT / f"{arm}_s{seed}"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _save_stats(vec_env, path: Path):
    with open(path, "wb") as f:
        pickle.dump({"obs_rms": vec_env.obs_rms, "ret_rms": vec_env.ret_rms,
                     "clip_obs": vec_env.clip_obs, "norm_obs": vec_env.norm_obs}, f)


def train(arm: str, seed: int, steps: int = STEPS, ckpt_every: int = CKPT_EVERY) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from flip7.agents.q_agent import QAgent
    from flip7.environments.gym_env import Flip7Env
    from flip7.experiments.curriculum import BASELINE, ensure_baseline

    torch.set_num_threads(1)
    ensure_baseline()
    variant = ARMS[arm]
    out = run_dir(arm, seed)
    out.mkdir(parents=True, exist_ok=True)
    q_agent = QAgent.load(str(BASELINE / "q_table.pkl"))

    env = DummyVecEnv([lambda: Flip7Env(opponent_agent=q_agent, obs_variant=variant)
                       for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=1e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=0, seed=seed * 1000, device="cpu",
    )

    class Ckpt(BaseCallback):
        """Every ckpt_every steps (< steps): model + its own VecNormalize stats."""

        def _on_step(self) -> bool:
            t = self.num_timesteps
            if t % ckpt_every == 0 and t < steps:
                self.model.save(str(out / f"ckpt_{t}"))
                _save_stats(self.training_env, out / f"ckpt_{t}_vecnorm.pkl")
            return True

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=Ckpt())
    model.save(str(out / "final"))
    _save_stats(env, out / "final_vecnorm.pkl")
    meta = {"arm": arm, "variant": variant, "seed": seed, "requested_steps": steps,
            "actual_steps": int(model.num_timesteps), "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    return out


def checkpoints(arm, seed) -> list[tuple[int, str]]:
    """(steps, name) for every saved checkpoint, final last."""
    d = run_dir(arm, seed)
    ck = sorted((int(p.stem.split("_")[1]), p.stem) for p in d.glob("ckpt_*.zip"))
    meta = json.loads((d / "meta.json").read_text())
    return ck + [(meta["actual_steps"], "final")]


def load_candidate(arm, seed, name):
    from flip7.experiments.common import build_frozen_ppo
    d = run_dir(arm, seed)
    return build_frozen_ppo(str(d / f"{name}.zip"), str(d / f"{name}_vecnorm.pkl"),
                            obs_variant=ARMS[arm])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _threshold():
    from flip7.experiments.evaluate_endgame import _threshold as thr
    return thr()


def _ppo_best():
    from flip7.experiments.common import build_frozen_ppo
    from flip7.experiments.curriculum import BASELINE
    return build_frozen_ppo(str(BASELINE / "ppo_best.zip"), str(BASELINE / "ppo_best_vecnorm.pkl"))


def validate(arm: str, seed: int) -> dict:
    """Win rate vs ThresholdAgent on validation seeds for every checkpoint."""
    from flip7.experiments.common import paired_eval
    torch.set_num_threads(1)
    path = run_dir(arm, seed) / "val.json"
    if path.exists():
        return json.loads(path.read_text())
    thr = _threshold()
    curve = [{"steps": t, "name": name,
              "val": paired_eval(load_candidate(arm, seed, name), thr, VAL_SEEDS)["win_rate"]}
             for t, name in checkpoints(arm, seed)]
    best = max(curve, key=lambda c: (c["val"], c["steps"]))
    res = {"curve": curve, "selected": best["name"], "selected_steps": best["steps"]}
    path.write_text(json.dumps(res, indent=2))
    return res


def validate_q(args) -> dict:
    """Learning curve vs the training opponent (Q-agent), same validation seeds."""
    arm, seed = args
    from flip7.agents.q_agent import QAgent
    from flip7.experiments.common import paired_eval
    from flip7.experiments.curriculum import BASELINE
    torch.set_num_threads(1)
    path = run_dir(arm, seed) / "val_q.json"
    if path.exists():
        return json.loads(path.read_text())
    q = QAgent.load(str(BASELINE / "q_table.pkl"))
    curve = [{"steps": t, "name": name,
              "val": paired_eval(load_candidate(arm, seed, name), q, VAL_SEEDS)["win_rate"]}
             for t, name in checkpoints(arm, seed)]
    path.write_text(json.dumps({"curve": curve}, indent=2))
    return {"curve": curve}


def curves_vs_q(procs: int = 4):
    jobs = [(a, s) for s in SEEDS for a in ARMS]
    with Pool(procs) as pool:
        pool.map(validate_q, jobs)
    ref = RESULTS / "features_reference.json"
    data = json.loads(ref.read_text())
    if "val_q" not in data["ppo_best"]:
        from flip7.agents.q_agent import QAgent
        from flip7.experiments.common import paired_eval
        from flip7.experiments.curriculum import BASELINE
        q = QAgent.load(str(BASELINE / "q_table.pkl"))
        data["ppo_best"]["val_q"] = paired_eval(_ppo_best(), q, VAL_SEEDS)["win_rate"]
        ref.write_text(json.dumps(data, indent=2))


class MistakeLogger(Agent):
    """Counts decisions that go against a clear one-step EV (|E(draw)| > 3)."""

    def __init__(self, agent):
        self.agent = agent
        self.counts = defaultdict(lambda: [0, 0, 0, 0])  # bucket -> [neg_n, neg_draw, pos_n, pos_stay]

    def choose_action(self, player, opponent, game):
        a = int(self.agent.choose_action(player, opponent, game))
        ev = expected_draw_gain(player, game)
        left = game.deck.remaining()
        b = next(name for name, lo, hi in DECK_BUCKETS if lo <= left < hi)
        c = self.counts[b]
        if ev < -EV_CLEAR:
            c[0] += 1
            c[1] += a == 1
        elif ev > EV_CLEAR:
            c[2] += 1
            c[3] += a == 0
        return a

    def summary(self) -> dict:
        out = {}
        tot = [0, 0, 0, 0]
        for name, _, _ in DECK_BUCKETS:
            c = self.counts[name]
            tot = [x + y for x, y in zip(tot, c)]
            out[name] = {"wrong_draw": c[1] / max(c[0], 1), "wrong_stay": c[3] / max(c[2], 1),
                         "n_clear": c[0] + c[2]}
        out["all"] = {"wrong_draw": tot[1] / max(tot[0], 1), "wrong_stay": tot[3] / max(tot[2], 1),
                      "mistake_rate": (tot[1] + tot[3]) / max(tot[0] + tot[2], 1),
                      "n_clear": tot[0] + tot[2]}
        return out


def test_model(cand) -> dict:
    """Held-out test metrics for one candidate agent."""
    from flip7.agents.q_agent import QAgent
    from flip7.experiments.common import paired_eval
    from flip7.experiments.curriculum import BASELINE
    from flip7.experiments.evaluate_endgame import (
        CHECK_SEEDS, P1_SEEDS, SeededRandom, paired_eval_seeded, suite_eval)
    thr = _threshold()
    logged = MistakeLogger(cand)
    p1 = paired_eval(logged, thr, P1_SEEDS)["win_rate"]
    with open(RESULTS / "endgame_positions.pkl", "rb") as f:
        suite = pickle.load(f)
    q = QAgent.load(str(BASELINE / "q_table.pkl"))
    return {
        "P1": p1,
        "mistakes": logged.summary(),
        "P2": suite_eval(cand, thr, suite)["win_rate"],
        "h2h_ppo_best": paired_eval(cand, _ppo_best(), CHECK_SEEDS)["win_rate"],
        "q_agent": paired_eval(cand, q, CHECK_SEEDS)["win_rate"],
        "random": paired_eval_seeded(cand, SeededRandom(), CHECK_SEEDS)["win_rate"],
    }


def test_run(arm: str, seed: int) -> dict:
    torch.set_num_threads(1)
    path = run_dir(arm, seed) / "test.json"
    if path.exists():
        return json.loads(path.read_text())
    val = validate(arm, seed)
    res = {"final": test_model(load_candidate(arm, seed, "final")), "selected_name": val["selected"]}
    res["selected"] = (res["final"] if val["selected"] == "final"
                       else test_model(load_candidate(arm, seed, val["selected"])))
    path.write_text(json.dumps(res, indent=2))
    return res


def reference_eval() -> dict:
    """ppo_best (and ThresholdAgent's own mistake rate = 0 by construction) on the same seeds."""
    torch.set_num_threads(1)
    path = RESULTS / "features_reference.json"
    if path.exists():
        return json.loads(path.read_text())
    from flip7.experiments.common import paired_eval
    best = _ppo_best()
    res = {"ppo_best": {"val": paired_eval(best, _threshold(), VAL_SEEDS)["win_rate"],
                        **test_model(best)}}
    path.write_text(json.dumps(res, indent=2))
    return res


def _job(kind_arm_seed):
    kind, arm, seed = kind_arm_seed
    if kind == "train":
        if not (run_dir(arm, seed) / "meta.json").exists():
            train(arm, seed)
        return kind, arm, seed, None
    if kind == "eval":
        test_run(arm, seed)
        return kind, arm, seed, validate(arm, seed)["selected"]
    if kind == "ref":
        reference_eval()
        return kind, arm, seed, None


def run_all(procs: int = 4):
    # Seed-major order so a partial run still has every arm at the same seeds.
    jobs = [("train", a, s) for s in SEEDS for a in ARMS]
    with Pool(procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, jobs):
            print("done", r, time.strftime("%H:%M:%S"), flush=True)
    jobs = [("ref", None, None)] + [("eval", a, s) for s in SEEDS for a in ARMS]
    with Pool(procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, jobs):
            print("done", r, time.strftime("%H:%M:%S"), flush=True)


# ---------------------------------------------------------------------------
# Round-robin and report
# ---------------------------------------------------------------------------

def _rr_pair(args):
    from flip7.experiments.common import paired_eval
    torch.set_num_threads(1)
    (na, spec_a), (nb, spec_b) = args
    return na, nb, paired_eval(_build(spec_a), _build(spec_b), RR_SEEDS)["win_rate"]


def _build(spec):
    if spec == "ppo_best":
        return _ppo_best()
    if spec == "threshold":
        return _threshold()
    arm, seed, name = spec
    return load_candidate(arm, seed, name)


def round_robin(procs: int = 4) -> dict:
    path = RESULTS / "features_round_robin.json"
    if path.exists():
        return json.loads(path.read_text())
    players = []
    for arm in ARMS:
        vals = {s: validate(arm, s) for s in SEEDS if (run_dir(arm, s) / "val.json").exists()}
        s = max(vals, key=lambda k: max(c["val"] for c in vals[k]["curve"]))
        players.append((f"{arm}_s{s}:{vals[s]['selected']}", (arm, s, vals[s]["selected"])))
    players += [("ppo_best", "ppo_best"), ("threshold", "threshold")]
    pairs = [(players[i], players[j]) for i in range(len(players)) for j in range(i + 1, len(players))]
    with Pool(procs) as pool:
        out = pool.map(_rr_pair, pairs)
    res = {"players": [p[0] for p in players], "matches": [list(o) for o in out]}
    path.write_text(json.dumps(res, indent=2))
    return res


def plot_curves(panels: list, path: Path):
    """panels: [(title, ylabel, {arm: {seed: val_dict}}, ref_value)]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, (title, ylabel, vals, ref_val) in zip(axes, panels):
        for arm, runs in vals.items():
            xs, ys = None, []
            for seed, v in sorted(runs.items()):
                x = [c["steps"] / 1e6 for c in v["curve"]]
                y = [c["val"] * 100 for c in v["curve"]]
                ax.plot(x, y, color=colors[arm], alpha=0.25, lw=1)
                xs = x
                ys.append(y)
            ax.plot(xs, np.mean(ys, axis=0), color=colors[arm], lw=2.5, marker="o", ms=4,
                    label=f"{arm}: {ARMS[arm]} (mean of {len(ys)})")
        ax.axhline(ref_val * 100, color="grey", ls="--", lw=1.2, label="ppo_best (deployed)")
        ax.set_xlabel("training steps (millions)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def report():
    from flip7.experiments.evaluate_endgame import summarize, welch
    ref = reference_eval()["ppo_best"]
    vals = {a: {s: validate(a, s) for s in SEEDS if (run_dir(a, s) / "val.json").exists()} for a in ARMS}
    tests = {a: {s: json.loads((run_dir(a, s) / "test.json").read_text())
                 for s in SEEDS if (run_dir(a, s) / "test.json").exists()} for a in ARMS}
    panels = [("vs ThresholdAgent (held out)", "win rate (%), validation 500 seeds x 2", vals, ref["val"])]
    q_vals = {a: {s: json.loads((run_dir(a, s) / "val_q.json").read_text()) for s in SEEDS
                  if (run_dir(a, s) / "val_q.json").exists()} for a in ARMS}
    if all(q_vals.values()) and "val_q" in ref:
        panels.append(("vs Q-agent (training opponent)", "win rate (%), same seeds", q_vals, ref["val_q"]))
    plot_curves(panels, RESULTS / "features_learning_curves.png")

    metrics = ["P1", "P2", "h2h_ppo_best", "q_agent", "random", "mistake_rate"]

    def get(t, m):
        return t["mistakes"]["all"]["mistake_rate"] if m == "mistake_rate" else t[m]

    lines = ["# P(bust)/E(draw) feature ablation — results", "",
             "Arms: A = base43 (current), B = added46 (+P(bust), E(draw), cards left), "
             "C = compact12. Stage 1 recipe vs Q-agent, 2M steps, seeds "
             f"{SEEDS}. Units = runs. P1 = full games vs ThresholdAgent (2,500 seeds x 2).", "",
             "![learning curves](features_learning_curves.png)", ""]
    for which in ("final", "selected"):
        data = {a: {m: [get(tests[a][s][which], m) for s in sorted(tests[a])] for m in metrics} for a in ARMS}
        lines += [f"## {which.capitalize()} checkpoints", "",
                  "| metric | ppo_best | A mean [95% CI] | B mean [95% CI] | C mean [95% CI] | B−A (p) | C−B (95% CI) |",
                  "|---|---|---|---|---|---|---|"]
        for m in metrics:
            s = {a: summarize(data[a][m]) for a in ARMS}
            ba = s["B"]["mean"] - s["A"]["mean"]
            pba = welch(data["B"][m], data["A"][m], "two-sided")
            cb = _diff_ci(data["C"][m], data["B"][m])
            cells = " | ".join(f"{s[a]['mean']:.3f} [{s[a]['ci95'][0]:.3f}, {s[a]['ci95'][1]:.3f}]" for a in ARMS)
            lines.append(f"| {m} | {get(ref, m):.3f} | {cells} | {ba:+.3f} ({pba:.3g}) | "
                         f"{cb[0]:+.3f} [{cb[1]:+.3f}, {cb[2]:+.3f}] |")
        lines.append("")
        if which == "final":
            final = data
    # Pre-registered decisions on final checkpoints.
    A, B, C = (final[a] for a in ARMS)
    sA = summarize(A["P1"])
    ba = summarize(B["P1"])["mean"] - sA["mean"]
    p_ba = welch(B["P1"], A["P1"], "two-sided")
    features_help = ba > 0 and p_ba < 0.05
    cb = _diff_ci(C["P1"], B["P1"])
    c_as_good = cb[1] > -0.01

    def no_harm(X):
        return {m: (summarize(X[m])["mean"] - summarize(A[m])["mean"] >= -0.01
                    and welch(X[m], A[m], "less") >= 0.05) for m in ("q_agent", "random")}

    lines += ["## Pre-registered decisions (final checkpoints)", "",
              f"- Sanity: arm A P1 = {sA['mean']:.3f} vs ppo_best {ref['P1']:.3f}",
              f"- **B vs A on P1**: {ba:+.3f}, Welch p = {p_ba:.4g} → features help: **{features_help}**",
              f"- C − B on P1: {cb[0]:+.3f}, 95% CI [{cb[1]:+.3f}, {cb[2]:+.3f}] → C as good as B: **{c_as_good}**",
              f"- No-harm vs A: B {no_harm(B)}, C {no_harm(C)}", ""]
    lines += ["## Mistake rates by deck size (final checkpoints, mean over runs)", "",
              "Share of clear decisions (|E(draw)| > 3) taken the wrong way, from the P1 games.", "",
              "| model | bucket | wrong draws (EV < −3) | wrong stays (EV > 3) |", "|---|---|---|---|"]
    for name, _, _ in DECK_BUCKETS:
        lines.append(f"| ppo_best | {name} | {ref['mistakes'][name]['wrong_draw']:.1%} | "
                     f"{ref['mistakes'][name]['wrong_stay']:.1%} |")
    for a in ARMS:
        for name, _, _ in DECK_BUCKETS:
            wd = np.mean([tests[a][s]["final"]["mistakes"][name]["wrong_draw"] for s in tests[a]])
            ws = np.mean([tests[a][s]["final"]["mistakes"][name]["wrong_stay"] for s in tests[a]])
            lines.append(f"| {a} | {name} | {wd:.1%} | {ws:.1%} |")
    lines.append("")
    rr_path = RESULTS / "features_round_robin.json"
    if rr_path.exists():
        rr = json.loads(rr_path.read_text())
        names = rr["players"]
        grid = {(a, b): w for a, b, w in rr["matches"]}
        lines += ["## Round-robin (row's win rate vs column, 1,000 seeds x 2)", "",
                  "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for a in names:
            cells = []
            for b in names:
                if a == b:
                    cells.append("—")
                elif (a, b) in grid:
                    cells.append(f"{grid[(a, b)]:.3f}")
                else:
                    cells.append(f"{1 - grid[(b, a)]:.3f}")
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Per-run detail", "", "| run | selected ckpt | val (selected) | P1 final | P1 selected | train min |",
              "|---|---|---|---|---|---|"]
    for a in ARMS:
        for s in sorted(tests[a]):
            v = vals[a][s]
            sel_val = next(c["val"] for c in v["curve"] if c["name"] == v["selected"])
            meta = json.loads((run_dir(a, s) / "meta.json").read_text())
            lines.append(f"| {a}_s{s} | {v['selected']} | {sel_val:.3f} | {tests[a][s]['final']['P1']:.3f} | "
                         f"{tests[a][s]['selected']['P1']:.3f} | {meta['seconds'] / 60:.0f} |")
    (RESULTS / "features_ablation.md").write_text("\n".join(lines) + "\n")
    summary = {"features_help": features_help, "p_ba": p_ba, "diff_ba": ba, "c_minus_b": cb,
               "c_as_good": c_as_good, "no_harm_B": no_harm(B), "no_harm_C": no_harm(C)}
    (RESULTS / "features_ablation.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n".join(lines))


def _diff_ci(x, y):
    """Welch 95% CI for mean(x) - mean(y)."""
    from scipy import stats
    x, y = np.asarray(x), np.asarray(y)
    d = x.mean() - y.mean()
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = (vx + vy) ** 0.5
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    h = stats.t.ppf(0.975, df) * se
    return float(d), float(d - h), float(d + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--round-robin", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--curves-q", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    if args.train:
        train(args.train, args.seed, args.steps, args.ckpt_every)
    elif args.all:
        run_all(args.procs)
    elif args.round_robin:
        round_robin(args.procs)
    elif args.curves_q:
        curves_vs_q(args.procs)
    elif args.report:
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
