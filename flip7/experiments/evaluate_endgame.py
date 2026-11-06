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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", action="store_true")
    args = ap.parse_args()
    if args.headroom:
        headroom()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
