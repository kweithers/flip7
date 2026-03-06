/**
 * Flip 7 game engine — faithful TypeScript port of the Python engine.
 * Deck: value v appears v times (v=1..12), 78 cards total.
 */

export function buildFullDeck(): number[] {
	const cards: number[] = [];
	for (let v = 1; v <= 12; v++) {
		for (let i = 0; i < v; i++) {
			cards.push(v);
		}
	}
	return cards;
}

// Seeded RNG (simple mulberry32)
export function createRng(seed: number): () => number {
	let s = seed | 0;
	return () => {
		s = (s + 0x6d2b79f5) | 0;
		let t = Math.imul(s ^ (s >>> 15), 1 | s);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function shuffle(arr: number[], rng: () => number): void {
	for (let i = arr.length - 1; i > 0; i--) {
		const j = Math.floor(rng() * (i + 1));
		[arr[i], arr[j]] = [arr[j], arr[i]];
	}
}

export class Deck {
	private cards: number[];
	private rng: () => number;

	constructor(rng: () => number) {
		this.rng = rng;
		this.cards = buildFullDeck();
		shuffle(this.cards, this.rng);
	}

	draw(cardsInPlay?: number[]): number {
		if (this.cards.length === 0) {
			this.reshuffle(cardsInPlay || []);
		}
		return this.cards.pop()!;
	}

	reshuffle(cardsInPlay: number[]): void {
		const inPlayCounts = new Map<number, number>();
		for (const c of cardsInPlay) {
			inPlayCounts.set(c, (inPlayCounts.get(c) || 0) + 1);
		}

		const fullCounts = new Map<number, number>();
		for (const c of buildFullDeck()) {
			fullCounts.set(c, (fullCounts.get(c) || 0) + 1);
		}

		this.cards = [];
		for (const [value, count] of fullCounts) {
			const available = count - (inPlayCounts.get(value) || 0);
			for (let i = 0; i < available; i++) {
				this.cards.push(value);
			}
		}
		shuffle(this.cards, this.rng);
	}

	remaining(): number {
		return this.cards.length;
	}

	cardCounts(): Map<number, number> {
		const counts = new Map<number, number>();
		for (let v = 1; v <= 12; v++) counts.set(v, 0);
		for (const c of this.cards) {
			counts.set(c, (counts.get(c) || 0) + 1);
		}
		return counts;
	}
}

export interface PlayerState {
	hand: number[];
	totalScore: number;
	busted: boolean;
	stayed: boolean;
}

export function createPlayer(): PlayerState {
	return { hand: [], totalScore: 0, busted: false, stayed: false };
}

export function handSum(player: PlayerState): number {
	return player.hand.reduce((a, b) => a + b, 0);
}

export function wouldBust(player: PlayerState, card: number): boolean {
	return player.hand.includes(card);
}

export function addCard(player: PlayerState, card: number): boolean {
	const busted = wouldBust(player, card);
	player.hand.push(card);
	if (busted) player.busted = true;
	return busted;
}

export function stay(player: PlayerState): number {
	player.stayed = true;
	return handSum(player);
}

export function roundScore(player: PlayerState): number {
	return player.busted ? 0 : handSum(player);
}

export function resetRound(player: PlayerState): void {
	player.hand = [];
	player.busted = false;
	player.stayed = false;
}

export function isDone(player: PlayerState): boolean {
	return player.busted || player.stayed;
}

export const WINNING_SCORE = 200;

export interface GameResult {
	winner: number; // 0 or 1
	scores: [number, number];
}

export class Game {
	deck: Deck;
	players: [PlayerState, PlayerState];
	currentPlayerIdx: number;
	roundNumber: number;
	gameOver: boolean;
	result: GameResult | null;
	private rng: () => number;

	constructor(seed: number) {
		this.rng = createRng(seed);
		this.deck = new Deck(this.rng);
		this.players = [createPlayer(), createPlayer()];
		this.currentPlayerIdx = 0;
		this.roundNumber = 0;
		this.gameOver = false;
		this.result = null;
	}

	private cardsInPlay(): number[] {
		return [...this.players[0].hand, ...this.players[1].hand];
	}

	startRound(): void {
		this.roundNumber++;
		for (const p of this.players) resetRound(p);
		for (const p of this.players) {
			const card = this.deck.draw(this.cardsInPlay());
			addCard(p, card);
		}
	}

	getCurrentPlayer(): PlayerState {
		return this.players[this.currentPlayerIdx];
	}

	getOpponent(): PlayerState {
		return this.players[1 - this.currentPlayerIdx];
	}

	step(action: 0 | 1): boolean {
		const player = this.getCurrentPlayer();

		if (action === 1) {
			// DRAW
			const card = this.deck.draw(this.cardsInPlay());
			addCard(player, card);
		} else {
			// STAY
			stay(player);
		}

		if (this.players.every(isDone)) {
			this.endRound();
			return true;
		}

		this.advanceTurn();
		return false;
	}

	private advanceTurn(): void {
		this.currentPlayerIdx = 1 - this.currentPlayerIdx;
		if (isDone(this.players[this.currentPlayerIdx])) {
			this.currentPlayerIdx = 1 - this.currentPlayerIdx;
		}
	}

	private endRound(): void {
		for (const p of this.players) {
			p.totalScore += roundScore(p);
		}

		const p0Over = this.players[0].totalScore >= WINNING_SCORE;
		const p1Over = this.players[1].totalScore >= WINNING_SCORE;

		if (p0Over || p1Over) {
			this.gameOver = true;
			let winner: number;
			if (p0Over && p1Over) {
				winner = this.players[0].totalScore >= this.players[1].totalScore ? 0 : 1;
			} else {
				winner = p0Over ? 0 : 1;
			}
			this.result = {
				winner,
				scores: [this.players[0].totalScore, this.players[1].totalScore]
			};
		}
	}

	isRoundOver(): boolean {
		return this.players.every(isDone);
	}
}
