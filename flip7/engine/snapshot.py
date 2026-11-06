"""Capture and restore a game at a round boundary (before ``start_round``).

A snapshot holds everything needed to replay the rest of the game exactly:
scores, round counter, who acts first, the remaining deck order, and the RNG
state (the deck shares ``Game.rng``, which drives reshuffles).
"""

from flip7.engine.game import Game


def capture(game: Game) -> dict:
    """Snapshot a game between rounds (hands are about to be reset)."""
    return {
        "scores": [p.total_score for p in game.players],
        "round_number": game.round_number,
        "current_player_idx": game.current_player_idx,
        "deck_cards": list(game.deck._cards),
        "rng_state": game.rng.getstate(),
    }


def restore(snap: dict, swap_seats: bool = False) -> Game:
    """Rebuild a Game from a snapshot.

    Seats keep their scores and deal order. ``swap_seats`` is not supported on
    purpose: to evaluate an agent in the other seat, pass the agents to
    ``play_game`` in the other order instead.
    """
    if swap_seats:
        raise ValueError("swap agents, not seats")
    game = Game(seed=0)
    # Restore state on the existing rng object: the deck holds a reference to it.
    game.rng.setstate(snap["rng_state"])
    game.deck._cards = list(snap["deck_cards"])
    for p, score in zip(game.players, snap["scores"]):
        p.total_score = score
    game.round_number = snap["round_number"]
    game.current_player_idx = snap["current_player_idx"]
    return game
