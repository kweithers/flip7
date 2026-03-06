import random
from collections import Counter

from flip7.engine.deck import Deck, build_full_deck


def test_full_deck_has_78_cards():
    assert len(build_full_deck()) == 78


def test_full_deck_value_counts():
    counts = Counter(build_full_deck())
    for v in range(1, 13):
        assert counts[v] == v


def test_draw_reduces_remaining():
    deck = Deck(rng=random.Random(42))
    assert deck.remaining() == 78
    deck.draw()
    assert deck.remaining() == 77


def test_card_counts_match_remaining():
    deck = Deck(rng=random.Random(42))
    # Draw a few cards
    for _ in range(10):
        deck.draw()
    counts = deck.card_counts()
    assert sum(counts.values()) == deck.remaining()


def test_reshuffle_excludes_cards_in_play():
    deck = Deck(rng=random.Random(42))
    # Draw all cards
    drawn = []
    for _ in range(78):
        drawn.append(deck.draw())
    assert deck.remaining() == 0

    # Reshuffle excluding some cards
    cards_in_play = [1, 2, 2, 3]  # 4 cards in play
    deck.reshuffle(cards_in_play)
    assert deck.remaining() == 78 - 4

    # Verify the reshuffled deck has correct counts
    counts = deck.card_counts()
    # Value 1: 1 total - 1 in play = 0
    assert counts[1] == 0
    # Value 2: 2 total - 2 in play = 0
    assert counts[2] == 0
    # Value 3: 3 total - 1 in play = 2
    assert counts[3] == 2


def test_draw_triggers_reshuffle_when_empty():
    deck = Deck(rng=random.Random(42))
    # Draw all 78 cards
    for _ in range(78):
        deck.draw()
    assert deck.is_empty()

    # Drawing again should trigger reshuffle
    cards_in_play = [1]
    card = deck.draw(cards_in_play)
    assert card in range(1, 13)
    assert deck.remaining() == 78 - 1 - 1  # 78 - 1 (drawn just now) - 1 (in play)
