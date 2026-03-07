/**
 * Q-learning agent for Flip 7.
 * Loads a Q-table (JSON) and makes greedy decisions based on hand card presence.
 */

import type { PlayerState } from './gameEngine';

type QTable = Record<string, [number, number]>;

export class QAgent {
	private qTable: QTable = {};

	async init(url: string): Promise<void> {
		const response = await fetch(url);
		this.qTable = await response.json();
	}

	private getState(player: PlayerState): string {
		const presence = new Array(12).fill(0);
		for (const card of player.hand) {
			presence[card - 1] = 1;
		}
		return presence.join(',');
	}

	chooseAction(player: PlayerState): 0 | 1 {
		const state = this.getState(player);
		const qValues = this.qTable[state];
		if (!qValues) {
			// Unknown state — default to STAY
			return 0;
		}
		return qValues[1] > qValues[0] ? 1 : 0;
	}
}
