/**
 * ONNX Runtime Web agent for Flip 7.
 * Loads a trained PPO model and normalization stats,
 * builds 41-dim observations, and returns STAY/DRAW decisions.
 */

import * as ort from 'onnxruntime-web';
import type { Game, PlayerState } from './gameEngine';

interface NormalizeStats {
	mean: number[];
	var: number[];
}

export class OnnxAgent {
	private session: ort.InferenceSession | null = null;
	private stats: NormalizeStats | null = null;

	async init(modelUrl: string, statsUrl: string): Promise<void> {
		this.session = await ort.InferenceSession.create(modelUrl, {
			executionProviders: ['wasm']
		});
		const response = await fetch(statsUrl);
		this.stats = await response.json();
	}

	buildObservation(
		player: PlayerState,
		opponent: PlayerState,
		game: Game
	): Float32Array {
		const obs = new Float32Array(41);
		const deckCounts = game.deck.cardCounts();

		// My hand presence (12)
		for (const card of player.hand) {
			obs[card - 1] = 1.0;
		}

		// My hand size (1)
		obs[12] = player.hand.length / 12.0;

		// My total score (1)
		obs[13] = Math.min(player.totalScore / 200.0, 1.0);

		// Opponent hand presence (12)
		for (const card of opponent.hand) {
			obs[14 + card - 1] = 1.0;
		}

		// Opponent hand size (1)
		obs[26] = opponent.hand.length / 12.0;

		// Opponent total score (1)
		obs[27] = Math.min(opponent.totalScore / 200.0, 1.0);

		// Opponent has stayed (1)
		obs[28] = opponent.stayed ? 1.0 : 0.0;

		// Deck counts (12)
		for (let v = 1; v <= 12; v++) {
			obs[29 + v - 1] = (deckCounts.get(v) || 0) / v;
		}

		return obs;
	}

	normalizeObs(obs: Float32Array): Float32Array {
		if (!this.stats) return obs;
		const normalized = new Float32Array(obs.length);
		for (let i = 0; i < obs.length; i++) {
			normalized[i] = (obs[i] - this.stats.mean[i]) / Math.sqrt(this.stats.var[i] + 1e-8);
			normalized[i] = Math.max(-10, Math.min(10, normalized[i]));
		}
		return normalized;
	}

	async chooseAction(
		player: PlayerState,
		opponent: PlayerState,
		game: Game
	): Promise<0 | 1> {
		if (!this.session) throw new Error('Agent not initialized');

		const obs = this.buildObservation(player, opponent, game);
		const normalizedObs = this.normalizeObs(obs);

		const tensor = new ort.Tensor('float32', normalizedObs, [1, 41]);
		const results = await this.session.run({ observation: tensor });
		const logits = results['action_logits'].data as Float32Array;

		// argmax: 0=STAY, 1=DRAW
		return logits[1] > logits[0] ? 1 : 0;
	}
}
