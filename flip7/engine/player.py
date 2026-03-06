class Player:
    def __init__(self):
        self.hand: list[int] = []
        self.total_score: int = 0
        self.busted: bool = False
        self.stayed: bool = False

    def hand_sum(self) -> int:
        return sum(self.hand)

    def unique_values(self) -> set[int]:
        return set(self.hand)

    def would_bust(self, card: int) -> bool:
        return card in self.unique_values()

    def add_card(self, card: int) -> bool:
        """Add a card to hand. Returns True if busted (duplicate)."""
        if self.would_bust(card):
            self.hand.append(card)
            self.busted = True
            return True
        self.hand.append(card)
        return False

    def stay(self) -> int:
        """Stay and bank points. Returns the round score."""
        self.stayed = True
        return self.hand_sum()

    def round_score(self) -> int:
        """Score for this round: 0 if busted, hand sum otherwise."""
        if self.busted:
            return 0
        return self.hand_sum()

    def reset_round(self) -> None:
        """Reset hand state for a new round. Score carries over."""
        self.hand = []
        self.busted = False
        self.stayed = False

    @property
    def is_done(self) -> bool:
        """Player is done for this round (busted or stayed)."""
        return self.busted or self.stayed
