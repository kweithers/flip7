"""End-game curriculum experiment: Step 0 (calibration, headroom, capture).

Usage:
    uv run python -m flip7.experiments.evaluate_endgame --headroom
"""

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

from flip7.agents.base import Agent
from flip7.agents.threshold_agent import ThresholdAgent
from flip7.engine.game import Game
from flip7.engine.snapshot import capture
from flip7.experiments.common import (
    CLOSE_MIN,
    build_frozen_ppo,
    is_close_endgame,
    paired_eval,
)

RESULTS = Path("results")
BASELINE_MODEL = "models/ppo_best.zip"
BASELINE_STATS = "models/ppo_best_vecnorm.pkl"

CALIB_SEEDS = range(0, 1000)
CAPTURE_SEEDS = range(100_000, 102_500)
BURN_SEEDS = range(200_000, 201_000)
CALIB_TARGET = (0.55, 0.85)
AGGRESSION_GRID = [0.2, 0.0, 0.5, 1.0, 2.0]
SUITE_MAX = 2000
BANDS = [(150, 175), (175, 190), (190, 200)]


class DecisionLogger(Agent):
    """Wraps an agent and logs (hand, deck size, scores, action) per decision."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.log: list[tuple] = []

    def choose_action(self, player, opponent, game):
        action = self.agent.choose_action(player, opponent, game)
        self.log.append((
            tuple(sorted(set(player.hand))),
            game.deck.remaining(),
            player.total_score,
            opponent.total_score,
            int(action),
        ))
        return action


def play_with_capture(agents, seed):
    """Play a full game, snapshotting every close end-game round boundary."""
    game = Game(seed=seed)
    positions = []
    while not game.game_over:
        scores = [p.total_score for p in game.players]
        if is_close_endgame(scores):
            positions.append(capture(game))
        game.play_round(agents)
        game.current_player_idx = game.round_number % 2
    return game.result, positions


def calibrate(ppo) -> tuple[float, list]:
    tried = []
    for a in AGGRESSION_GRID:
        r = paired_eval(ppo, ThresholdAgent(aggression=a), CALIB_SEEDS)
        tried.append((a, r["win_rate"]))
        if CALIB_TARGET[0] <= r["win_rate"] <= CALIB_TARGET[1]:
            return a, tried
    # Nothing inside the window: keep the default (documented deviation).
    return AGGRESSION_GRID[0], tried


def score_awareness(log) -> dict:
    """Draw rate trailing-near-200 minus leading-near-200 on matched states."""
    groups = {"trail": defaultdict(list), "lead": defaultdict(list)}
    for hand, deck, me, opp, act in log:
        key = (hand, deck // 10)
        if opp >= CLOSE_MIN and opp - me >= 10:
            groups["trail"][key].append(act)
        elif me >= CLOSE_MIN and me - opp >= 10:
            groups["lead"][key].append(act)
    num = den = 0.0
    for key in groups["trail"].keys() & groups["lead"].keys():
        t, l = groups["trail"][key], groups["lead"][key]
        w = min(len(t), len(l))
        num += w * (sum(t) / len(t) - sum(l) / len(l))
        den += w
    return {"diff": num / den if den else float("nan"), "matched_weight": den}


def leader_band(snap):
    top = max(snap["scores"])
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= top < hi:
            return i
    raise ValueError(snap["scores"])


def headroom():
    RESULTS.mkdir(exist_ok=True)
    ppo = build_frozen_ppo(BASELINE_MODEL, BASELINE_STATS)

    # 1. Calibrate and freeze ThresholdAgent.
    aggression, tried = calibrate(ppo)
    thr = ThresholdAgent(aggression=aggression)

    # 2. Capture games.
    logger = DecisionLogger(ppo)
    n_games = n_close = n_close_lost = n_lost = 0
    per_game_positions = []
    for s in CAPTURE_SEEDS:
        for seat in (0, 1):
            agents = [logger, thr] if seat == 0 else [thr, logger]
            result, positions = play_with_capture(agents, s)
            n_games += 1
            lost = result.winner != seat
            n_lost += lost
            if positions:
                n_close += 1
                n_close_lost += lost
                per_game_positions.append((s, seat, positions))

    # Burn distribution from separate games.
    burn_deck_sizes = []
    for s in BURN_SEEDS:
        seat = s % 2
        agents = [ppo, thr] if seat == 0 else [thr, ppo]
        _, positions = play_with_capture(agents, s)
        burn_deck_sizes += [len(p["deck_cards"]) for p in positions]

    # 3-4. Headroom and score-awareness.
    f_close = n_close / n_games
    raw = n_close_lost / n_games
    bound = 0.3 * raw
    mde_full = 0.03 * f_close
    go = bound >= 2 * mde_full
    aware = score_awareness(logger.log)

    # 6. Real-position suite: one random close position per game, stratified by band.
    rng = random.Random(12345)
    by_band = defaultdict(list)
    for s, seat, positions in per_game_positions:
        snap = dict(rng.choice(positions))
        snap.update(source_seed=s, ppo_seat=seat)
        by_band[leader_band(snap)].append(snap)
    per_band = SUITE_MAX // len(BANDS)
    suite = []
    for b in range(len(BANDS)):
        items = by_band[b]
        rng.shuffle(items)
        suite += items[:per_band]

    with open(RESULTS / "endgame_positions.pkl", "wb") as f:
        pickle.dump(suite, f)
    with open(RESULTS / "burn_deck_sizes.json", "w") as f:
        json.dump(burn_deck_sizes, f)

    summary = {
        "threshold_aggression": aggression,
        "calibration_tried": tried,
        "calibration_in_window": CALIB_TARGET[0] <= dict(tried)[aggression] <= CALIB_TARGET[1],
        "games": n_games,
        "ppo_win_rate_vs_threshold": 1 - n_lost / n_games,
        "f_close": f_close,
        "close_lost_fraction_of_games": raw,
        "close_fraction_of_losses": n_close_lost / max(n_lost, 1),
        "headroom_bound": bound,
        "mde_full_game_equivalent": mde_full,
        "go": go,
        "score_awareness": aware,
        "suite_size": len(suite),
        "suite_per_band": {f"{lo}-{hi - 1}": min(len(by_band[i]), per_band)
                           for i, (lo, hi) in enumerate(BANDS)},
        "burn_samples": len(burn_deck_sizes),
        "burn_deck_size_quartiles": _quartiles(burn_deck_sizes),
    }
    with open(RESULTS / "endgame_headroom.json", "w") as f:
        json.dump(summary, f, indent=2)
    _write_md(summary)
    print(json.dumps(summary, indent=2))
    return summary


def _quartiles(xs):
    if not xs:
        return []
    xs = sorted(xs)
    return [xs[int(q * (len(xs) - 1))] for q in (0, 0.25, 0.5, 0.75, 1)]


def _write_md(s):
    pct = lambda x: f"{100 * x:.1f}%"
    lines = [
        "# End-game experiment — Step 0: headroom",
        "",
        f"- ThresholdAgent aggression (frozen): **{s['threshold_aggression']}**"
        f" (calibration tried: {', '.join(f'{a}→{pct(w)}' for a, w in s['calibration_tried'])};"
        f" in 55–85% window: {s['calibration_in_window']})",
        f"- ppo_best win rate vs ThresholdAgent ({s['games']} games): {pct(s['ppo_win_rate_vs_threshold'])}",
        f"- Games reaching a close end-game (`f_close`): {pct(s['f_close'])}",
        f"- Games that reach a close end-game and ppo_best loses: {pct(s['close_lost_fraction_of_games'])}"
        f" ({pct(s['close_fraction_of_losses'])} of its losses)",
        f"- Headroom bound (×0.3): **{pct(s['headroom_bound'])}** of full-game win rate",
        f"- MDE full-game equivalent (3 pp × f_close): {pct(s['mde_full_game_equivalent'])};"
        f" gate requires bound ≥ 2× that",
        f"- Score-awareness (draw rate trailing − leading, matched states):"
        f" {pct(s['score_awareness']['diff'])} (weight {s['score_awareness']['matched_weight']:.0f}) — informational",
        f"- Real-position suite: {s['suite_size']} positions, per band {s['suite_per_band']}",
        f"- Burn distribution: {s['burn_samples']} samples, deck-size quartiles {s['burn_deck_size_quartiles']}",
        "",
        f"**Go/no-go: {'GO' if s['go'] else 'NO-GO'}**",
    ]
    (RESULTS / "endgame_headroom.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Evaluation of fine-tuned candidates
# ---------------------------------------------------------------------------

P1_SEEDS = range(300_000, 302_500)        # 2,500 seeds x 2 seats
CHECK_SEEDS = range(400_000, 401_000)     # 1,000 seeds x 2 seats
FROZEN_AGGRESSION_PATH = RESULTS / "endgame_headroom.json"


def _threshold():
    a = json.loads(FROZEN_AGGRESSION_PATH.read_text())["threshold_aggression"]
    return ThresholdAgent(aggression=a)


def _situation(cand_score, opp_score):
    d = cand_score - opp_score
    return "leading" if d > 10 else "trailing" if d < -10 else "close"


def suite_eval(candidate, opponent, suite) -> dict:
    """P2: play each saved position to the end, candidate in seat 0 then seat 1.

    Seats keep their scores and deal order; the candidate takes that seat's
    score, so each playthrough is labelled from the candidate's perspective.
    """
    from flip7.engine.snapshot import restore
    wins = games = 0
    by_type = defaultdict(lambda: [0, 0])
    by_band = defaultdict(lambda: [0, 0])
    for snap in suite:
        band = leader_band(snap)
        for seat in (0, 1):
            game = restore(snap)
            agents = [candidate, opponent] if seat == 0 else [opponent, candidate]
            won = game.play_game(agents).winner == seat
            kind = _situation(snap["scores"][seat], snap["scores"][1 - seat])
            wins += won
            games += 1
            by_type[kind][0] += won
            by_type[kind][1] += 1
            by_band[band][0] += won
            by_band[band][1] += 1
    return {
        "win_rate": wins / games, "games": games,
        "by_type": {k: w / n for k, (w, n) in by_type.items()},
        "by_band": {f"{BANDS[b][0]}-{BANDS[b][1] - 1}": w / n for b, (w, n) in sorted(by_band.items())},
    }


class SeededRandom(Agent):
    """RandomAgent whose RNG is reset per game seed, so seat-swapped pairs match."""

    def __init__(self, draw_prob=0.5):
        from flip7.agents.random_agent import RandomAgent
        self.inner = RandomAgent(draw_prob)

    def reseed(self, seed):
        self.inner.rng = random.Random(seed)

    def choose_action(self, player, opponent, game):
        return self.inner.choose_action(player, opponent, game)


def paired_eval_seeded(candidate, opponent: SeededRandom, seeds) -> dict:
    wins = games = 0
    for s in seeds:
        for seat in (0, 1):
            opponent.reseed(s)
            game = Game(seed=s)
            agents = [candidate, opponent] if seat == 0 else [opponent, candidate]
            wins += game.play_game(agents).winner == seat
            games += 1
    return {"wins": wins, "games": games, "win_rate": wins / games}


def evaluate_candidate(model_path: str, out_path: Path | None = None) -> dict:
    from flip7.agents.q_agent import QAgent
    import torch
    from flip7.experiments.curriculum import BASELINE, ensure_baseline
    torch.set_num_threads(1)
    ensure_baseline()
    stats = str(BASELINE / "ppo_best_vecnorm.pkl")  # frozen normalization for every candidate
    cand = build_frozen_ppo(model_path, stats)
    thr = _threshold()
    with open(RESULTS / "endgame_positions.pkl", "rb") as f:
        suite = pickle.load(f)
    best = build_frozen_ppo(str(BASELINE / "ppo_best.zip"), stats)
    final = build_frozen_ppo(str(BASELINE / "ppo_final.zip"), str(BASELINE / "vec_normalize_final.pkl"))
    q = QAgent.load(str(BASELINE / "q_table.pkl"))

    res = {
        "model": model_path,
        "P2_suite_vs_threshold": suite_eval(cand, thr, suite),
        "P1_full_vs_threshold": paired_eval(cand, thr, P1_SEEDS)["win_rate"],
        "q_agent": paired_eval(cand, q, CHECK_SEEDS)["win_rate"],
        "random": paired_eval_seeded(cand, SeededRandom(), CHECK_SEEDS)["win_rate"],
        "suite_vs_ppo_best": suite_eval(cand, best, suite)["win_rate"],
        "h2h_ppo_best": paired_eval(cand, best, CHECK_SEEDS)["win_rate"],
        "h2h_ppo_final": paired_eval(cand, final, CHECK_SEEDS)["win_rate"],
    }
    if out_path:
        out_path.write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# Statistics and reports
# ---------------------------------------------------------------------------

METRICS = ["P2", "P1", "q_agent", "random", "suite_vs_ppo_best", "h2h_ppo_best", "h2h_ppo_final"]


def _flat(res):
    return {
        "P2": res["P2_suite_vs_threshold"]["win_rate"], "P1": res["P1_full_vs_threshold"],
        **{k: res[k] for k in METRICS[2:]},
    }


def load_runs():
    from flip7.experiments.curriculum import FT
    runs = {"control": {}, "curriculum": {}}
    for d in sorted(FT.glob("*_s*")):
        ev = d / "eval.json"
        if not ev.exists():
            continue
        arm, seed = d.name.rsplit("_s", 1)
        meta = json.loads((d / "meta.json").read_text())
        runs[arm][int(seed)] = {**_flat(json.loads(ev.read_text())),
                                "scenario_step_share": meta["scenario_step_share"]}
    return runs


def summarize(xs):
    from scipy import stats
    xs = list(xs)
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": m, "sd": float("nan"), "ci95": [float("nan")] * 2}
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    h = stats.t.ppf(0.975, n - 1) * sd / n ** 0.5
    return {"n": n, "mean": m, "sd": sd, "ci95": [m - h, m + h]}


def permutation_p(a, b, alternative="greater"):
    """Exact permutation test on the difference in means (a - b)."""
    from itertools import combinations
    pooled = list(a) + list(b)
    na = len(a)
    obs = sum(a) / na - sum(b) / len(b)
    count = total = 0
    for idx in combinations(range(len(pooled)), na):
        sa = [pooled[i] for i in idx]
        sb = [pooled[i] for i in range(len(pooled)) if i not in idx]
        d = sum(sa) / na - sum(sb) / len(sb)
        total += 1
        if alternative == "greater":
            count += d >= obs - 1e-12
        else:
            count += abs(d) >= abs(obs) - 1e-12
    return count / total


def welch(a, b, alternative):
    from scipy import stats
    return float(stats.ttest_ind(a, b, equal_var=False, alternative=alternative).pvalue)


def required_n(sd, mde=0.03, alpha=0.05, power=0.8):
    """Seeds per arm for a two-sided two-sample t-test (iterated t-approximation)."""
    from scipy import stats
    n = 2
    for _ in range(100):
        df = max(2 * n - 2, 1)
        t_a, t_b = stats.t.ppf(1 - alpha / 2, df), stats.t.ppf(power, df)
        new = max(2, int(-(-2 * (t_a + t_b) ** 2 * sd ** 2 // mde ** 2)))
        if new == n:
            break
        n = new
    return n


def sd_upper80(sd, n):
    """Upper bound of a one-sided 80% CI for a population SD."""
    from scipy import stats
    return sd * ((n - 1) / stats.chi2.ppf(0.2, n - 1)) ** 0.5


def step0():
    path = RESULTS / "step0_eval.json"
    if not path.exists():
        from flip7.experiments.curriculum import BASELINE, ensure_baseline
        ensure_baseline()
        evaluate_candidate(str(BASELINE / "ppo_best.zip"), path)
    return _flat(json.loads(path.read_text()))


def pilot_report():
    s0 = step0()
    runs = load_runs()
    ctrl = runs["control"]
    out = {"step0": s0, "runs": runs}
    lines = ["# End-game experiment — pilot", "",
             f"Runs: control seeds {sorted(ctrl)}, curriculum seeds {sorted(runs['curriculum'])}", ""]
    lines += ["| metric | step 0 | control mean ± sd | curriculum mean ± sd |", "|---|---|---|---|"]
    for m in METRICS:
        c = summarize([r[m] for r in ctrl.values()])
        u = summarize([r[m] for r in runs["curriculum"].values()]) if runs["curriculum"] else None
        lines.append(f"| {m} | {s0[m]:.3f} | {c['mean']:.3f} ± {c['sd']:.3f} | "
                     + (f"{u['mean']:.3f} ± {u['sd']:.3f}" if u else "–") + " |")
    p1_ctrl = summarize([r["P1"] for r in ctrl.values()])
    degraded = p1_ctrl["mean"] < s0["P1"] - 0.02
    sd2 = summarize([r["P2"] for r in ctrl.values()])["sd"]
    sd2_up = sd_upper80(sd2, len(ctrl))
    n_req = required_n(sd2_up)
    sd1 = p1_ctrl["sd"]
    share = [r["scenario_step_share"] for r in runs["curriculum"].values()]
    out.update(degraded=degraded, p2_sd=sd2, p2_sd_upper80=sd2_up, n_required=n_req, p1_sd=sd1)
    lines += [
        "",
        f"- Degradation gate (control P1 below step 0 by > 2 pp): **{'FAIL' if degraded else 'pass'}**"
        f" (control {p1_ctrl['mean']:.3f} vs step 0 {s0['P1']:.3f})",
        f"- P2 seed-to-seed SD (control): {sd2:.4f}; upper 80% bound {sd2_up:.4f}",
        f"- Seeds per arm for 80% power at 3 pp MDE on P2: **{n_req}**"
        + (" → exceeds 20, stop and report" if n_req > 20 else ""),
        f"- P1 SD (control): {sd1:.4f}",
        f"- Curriculum scenario step share: {', '.join(f'{x:.3f}' for x in share) or '–'}",
    ]
    (RESULTS / "endgame_pilot.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "endgame_pilot.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))
    return out


def final_report():
    s0 = step0()
    runs = load_runs()
    seeds = sorted(set(runs["control"]) & set(runs["curriculum"]))
    c = {m: [runs["control"][s][m] for s in seeds] for m in METRICS}
    u = {m: [runs["curriculum"][s][m] for s in seeds] for m in METRICS}
    rows, checks = [], {}
    for m in METRICS:
        sc, su = summarize(c[m]), summarize(u[m])
        diff = su["mean"] - sc["mean"]
        p_greater = welch(u[m], c[m], "greater")
        p_less = welch(u[m], c[m], "less")
        rows.append((m, s0[m], sc, su, diff, welch(u[m], c[m], "two-sided"), p_greater, p_less))
        if m in ("P1", "q_agent", "random"):
            checks[m] = diff >= -0.01 and p_less >= 0.05
    p2_p = welch(u["P2"], c["P2"], "two-sided")
    p2_diff = summarize(u["P2"])["mean"] - summarize(c["P2"])["mean"]
    p2_perm = permutation_p(u["P2"], c["P2"], "two-sided") if len(seeds) <= 12 else None
    success = p2_diff > 0 and p2_p < 0.05 and all(checks.values())
    p1_sig = rows[1][4] > 0 and rows[1][5] < 0.05

    lines = ["# End-game curriculum experiment — results", "",
             f"Seeds per arm: {len(seeds)} ({seeds}). Units = runs (final checkpoint at ~500k steps).", "",
             "| metric | step 0 | control mean [95% CI] | curriculum mean [95% CI] | diff | p (two-sided) |",
             "|---|---|---|---|---|---|"]
    for m, s, sc, su, d, p2s, _, _ in rows:
        lines.append(f"| {m} | {s:.3f} | {sc['mean']:.3f} [{sc['ci95'][0]:.3f}, {sc['ci95'][1]:.3f}] | "
                     f"{su['mean']:.3f} [{su['ci95'][0]:.3f}, {su['ci95'][1]:.3f}] | {d:+.3f} | {p2s:.3f} |")
    lines += [
        "",
        f"- **Primary (P2)**: diff {p2_diff:+.3f}, Welch p = {p2_p:.4f}"
        + (f", permutation p = {p2_perm:.4f}" if p2_perm is not None else ""),
        f"- No-harm checks (diff ≥ −1 pp and not significantly worse): {checks}",
        f"- **Success: {success}**",
        f"- Full-game improvement claimed (P1 significant): {p1_sig}",
    ]
    (RESULTS / "endgame_ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", action="store_true")
    ap.add_argument("--eval-run", help="models/ft/<run> directory to evaluate")
    ap.add_argument("--step0", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.headroom:
        headroom()
    elif args.eval_run:
        d = Path(args.eval_run)
        print(json.dumps(evaluate_candidate(str(d / "final.zip"), d / "eval.json"), indent=2))
    elif args.step0:
        print(json.dumps(step0(), indent=2))
    elif args.pilot:
        pilot_report()
    elif args.full:
        final_report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
