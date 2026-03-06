import random
from collections import Counter


def build_full_deck() -> list[int]:
    """Build the standard 78-card Flip 7 deck: v copies of value v for v=1..12."""
    cards = []
    for v in range(1, 13):
        cards.extend([v] * v)
    return cards


class Deck:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self._cards: list[int] = build_full_deck()
        self.rng.shuffle(self._cards)

    def draw(self, cards_in_play: list[int] | None = None) -> int:
        """Draw one card. If the deck is empty, reshuffle excluding cards_in_play."""
        if not self._cards:
            self.reshuffle(cards_in_play or [])
        return self._cards.pop()

    def reshuffle(self, cards_in_play: list[int]) -> None:
        """Rebuild and shuffle the deck, excluding cards currently in play."""
        in_play_counts = Counter(cards_in_play)
        full_counts = Counter(build_full_deck())
        self._cards = []
        for value, count in full_counts.items():
            available = count - in_play_counts.get(value, 0)
            self._cards.extend([value] * available)
        self.rng.shuffle(self._cards)

    def remaining(self) -> int:
        return len(self._cards)

    def card_counts(self) -> dict[int, int]:
        """Count of each value remaining in the deck."""
        counts = Counter(self._cards)
        return {v: counts.get(v, 0) for v in range(1, 13)}

    def is_empty(self) -> bool:
        return len(self._cards) == 0
