from flip7.agents.random_agent import RandomAgent
from flip7.engine.game import Action, Game
from flip7.engine.player import Player


def test_player_bust_on_duplicate():
    p = Player()
    assert not p.add_card(3)
    assert not p.add_card(5)
    assert p.add_card(3)  # duplicate -> bust
    assert p.busted
    assert p.round_score() == 0


def test_player_stay_banks_score():
    p = Player()
    p.add_card(3)
    p.add_card(7)
    score = p.stay()
    assert score == 10
    assert p.stayed
    assert p.round_score() == 10


def test_player_reset_round():
    p = Player()
    p.add_card(5)
    p.total_score = 50
    p.stay()
    p.reset_round()
    assert p.hand == []
    assert not p.busted
    assert not p.stayed
    assert p.total_score == 50  # score carries over


def test_game_start_round_deals_one_card_each():
    game = Game(seed=42)
    game.start_round()
    assert len(game.players[0].hand) == 1
    assert len(game.players[1].hand) == 1


def test_game_round_ends_when_both_done():
    game = Game(seed=42)
    game.start_round()

    # Player 0 stays
    game.current_player_idx = 0
    round_over = game.step(Action.STAY)
    assert not round_over  # Player 1 still playing

    # Player 1 stays
    round_over = game.step(Action.STAY)
    assert round_over


def test_game_scores_accumulate():
    game = Game(seed=42)
    game.start_round()

    # Both players stay immediately (each has 1 card)
    game.step(Action.STAY)  # Player 0
    game.step(Action.STAY)  # Player 1

    assert game.players[0].total_score > 0
    assert game.players[1].total_score > 0


def test_full_game_completes():
    agents = [RandomAgent(draw_prob=0.5), RandomAgent(draw_prob=0.5)]
    game = Game(seed=42)
    result = game.play_game(agents)

    assert result is not None
    assert result.winner in (0, 1)
    assert max(result.scores) >= 200


def test_both_over_200_highest_wins():
    """If both players cross 200 in the same round, highest score wins."""
    game = Game(seed=42)
    # Manually set scores close to 200
    game.players[0].total_score = 195
    game.players[1].total_score = 195

    # Play rounds until game ends
    agents = [RandomAgent(draw_prob=0.6), RandomAgent(draw_prob=0.6)]
    while not game.game_over:
        game.play_round(agents)
        game.current_player_idx = game.round_number % 2

    assert game.result is not None
    assert max(game.result.scores) >= 200
    # Winner should have the highest score
    winner = game.result.winner
    loser = 1 - winner
    assert game.result.scores[winner] >= game.result.scores[loser]


def test_deck_persists_across_rounds():
    game = Game(seed=42)
    initial_remaining = game.deck.remaining()

    game.start_round()
    # 2 cards dealt
    after_deal = game.deck.remaining()
    assert after_deal == initial_remaining - 2

    # Both stay
    game.step(Action.STAY)
    game.step(Action.STAY)

    # Start next round — deck should still have the reduced count
    # (previous hand cards return implicitly since round ended)
    before_second = game.deck.remaining()
    game.start_round()
    assert game.deck.remaining() == before_second - 2
