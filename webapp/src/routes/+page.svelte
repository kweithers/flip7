<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Game,
		handSum,
		isDone,
		roundScore,
		WINNING_SCORE,
		type GameResult
	} from '$lib/gameEngine';
	import { OnnxAgent } from '$lib/onnxAgent';

	const PLAYER_IDX = 0;
	const AI_IDX = 1;

	let game: Game | null = $state(null);
	let agent = new OnnxAgent();
	let agentReady = $state(false);
	let playerTurn = $state(false);
	let roundLog: string[] = $state([]);
	let gameMessage = $state('Loading AI model...');
	let gameResult: GameResult | null = $state(null);
	let aiThinking = $state(false);

	let playerHand = $state<number[]>([]);
	let aiHand = $state<number[]>([]);
	let playerScore = $state(0);
	let aiScore = $state(0);
	let roundNum = $state(0);
	let playerBusted = $state(false);
	let playerStayed = $state(false);
	let aiBusted = $state(false);
	let aiStayed = $state(false);
	let playerHandSum = $state(0);
	let aiHandSum = $state(0);

	function syncState() {
		if (!game) return;
		playerHand = [...game.players[PLAYER_IDX].hand];
		aiHand = [...game.players[AI_IDX].hand];
		playerScore = game.players[PLAYER_IDX].totalScore;
		aiScore = game.players[AI_IDX].totalScore;
		roundNum = game.roundNumber;
		playerBusted = game.players[PLAYER_IDX].busted;
		playerStayed = game.players[PLAYER_IDX].stayed;
		aiBusted = game.players[AI_IDX].busted;
		aiStayed = game.players[AI_IDX].stayed;
		playerHandSum = handSum(game.players[PLAYER_IDX]);
		aiHandSum = handSum(game.players[AI_IDX]);
	}

	onMount(async () => {
		try {
			await agent.init('/ppo_flip7.onnx', '/normalize_stats.json');
			agentReady = true;
			gameMessage = 'AI loaded! Click "New Game" to start.';
		} catch (e) {
			gameMessage = `Failed to load AI model: ${e}. Playing with random AI.`;
			agentReady = false;
		}
	});

	function startNewGame() {
		const seed = Math.floor(Math.random() * 2147483647);
		game = new Game(seed);
		gameResult = null;
		roundLog = [];
		syncState();
		startNewRound();
	}

	function startNewRound() {
		if (!game || game.gameOver) return;

		game.currentPlayerIdx = game.roundNumber % 2;
		game.startRound();
		syncState();

		addLog(`--- Round ${game.roundNumber} ---`);
		addLog(`You drew: ${game.players[PLAYER_IDX].hand[0]}, AI drew: ${game.players[AI_IDX].hand[0]}`);

		if (game.currentPlayerIdx === AI_IDX) {
			playerTurn = false;
			runAiTurn();
		} else {
			playerTurn = true;
			gameMessage = 'Your turn! Draw or Stay?';
		}
	}

	async function runAiTurn() {
		if (!game || game.gameOver) return;
		aiThinking = true;

		while (game.currentPlayerIdx === AI_IDX && !game.isRoundOver()) {
			await sleep(500);

			let action: 0 | 1;
			if (agentReady) {
				action = await agent.chooseAction(
					game.players[AI_IDX],
					game.players[PLAYER_IDX],
					game
				);
			} else {
				action = Math.random() < 0.4 ? 1 : 0;
			}

			const roundOver = game.step(action);
			syncState();

			if (action === 1) {
				const lastCard = game.players[AI_IDX].hand[game.players[AI_IDX].hand.length - 1];
				if (game.players[AI_IDX].busted) {
					addLog(`AI drew ${lastCard} and BUSTED!`);
				} else {
					addLog(`AI drew ${lastCard}`);
				}
			} else {
				addLog(`AI stays with ${handSum(game.players[AI_IDX])} points`);
			}

			if (roundOver) {
				endRound();
				return;
			}
		}

		aiThinking = false;

		if (!game.isRoundOver() && !isDone(game.players[PLAYER_IDX])) {
			playerTurn = true;
			gameMessage = 'Your turn! Draw or Stay?';
		}
	}

	async function playerAction(action: 0 | 1) {
		if (!game || !playerTurn || game.gameOver) return;
		playerTurn = false;

		const roundOver = game.step(action);
		syncState();

		if (action === 1) {
			const lastCard = game.players[PLAYER_IDX].hand[game.players[PLAYER_IDX].hand.length - 1];
			if (game.players[PLAYER_IDX].busted) {
				addLog(`You drew ${lastCard} and BUSTED!`);
			} else {
				addLog(`You drew ${lastCard}`);
			}
		} else {
			addLog(`You stay with ${handSum(game.players[PLAYER_IDX])} points`);
		}

		if (roundOver) {
			endRound();
			return;
		}

		if (game.currentPlayerIdx === AI_IDX && !isDone(game.players[AI_IDX])) {
			await runAiTurn();
		} else if (!isDone(game.players[PLAYER_IDX])) {
			playerTurn = true;
			gameMessage = 'Your turn! Draw or Stay?';
		}
	}

	function endRound() {
		if (!game) return;
		aiThinking = false;

		const pScore = roundScore(game.players[PLAYER_IDX]);
		const aScore = roundScore(game.players[AI_IDX]);
		addLog(`Round result: You ${pScore}, AI ${aScore}`);
		addLog(`Totals: You ${game.players[PLAYER_IDX].totalScore}, AI ${game.players[AI_IDX].totalScore}`);
		syncState();

		if (game.gameOver) {
			gameResult = game.result;
			if (gameResult!.winner === PLAYER_IDX) {
				gameMessage = `You win! Final: ${gameResult!.scores[0]} - ${gameResult!.scores[1]}`;
			} else {
				gameMessage = `AI wins! Final: ${gameResult!.scores[0]} - ${gameResult!.scores[1]}`;
			}
		} else {
			gameMessage = 'Round over! Starting next round...';
			setTimeout(() => startNewRound(), 1500);
		}
	}

	function addLog(msg: string) {
		roundLog = [...roundLog, msg];
	}

	function sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}
</script>

<svelte:head>
	<title>Flip 7 - Play Against AI</title>
</svelte:head>

<main>
	<h1>Flip 7</h1>
	<p class="subtitle">Push your luck card game</p>

	{#if !game}
		<div class="start-screen">
			<p>{gameMessage}</p>
			<button onclick={() => startNewGame()} disabled={!agentReady && gameMessage.includes('Loading')}>
				New Game
			</button>
		</div>
	{:else}
		<div class="game-board">
			<div class="scores">
				<div class="score-card" class:winning={playerScore >= WINNING_SCORE}>
					<span class="label">You</span>
					<span class="value">{playerScore}</span>
				</div>
				<div class="round-info">Round {roundNum}</div>
				<div class="score-card" class:winning={aiScore >= WINNING_SCORE}>
					<span class="label">AI</span>
					<span class="value">{aiScore}</span>
				</div>
			</div>

			<div class="hand-section">
				<h2>
					AI's Hand
					{#if aiStayed}<span class="badge stayed">Stayed</span>{/if}
					{#if aiBusted}<span class="badge busted">Busted!</span>{/if}
					{#if aiThinking}<span class="badge thinking">Thinking...</span>{/if}
				</h2>
				<div class="cards">
					{#each aiHand as card}
						<div class="card" class:busted={aiBusted}>{card}</div>
					{/each}
				</div>
				<div class="hand-info">Sum: {aiHandSum}</div>
			</div>

			<div class="hand-section player">
				<h2>
					Your Hand
					{#if playerStayed}<span class="badge stayed">Stayed</span>{/if}
					{#if playerBusted}<span class="badge busted">Busted!</span>{/if}
				</h2>
				<div class="cards">
					{#each playerHand as card}
						<div class="card player-card" class:busted={playerBusted}>{card}</div>
					{/each}
				</div>
				<div class="hand-info">Sum: {playerHandSum}</div>
			</div>

			<div class="actions">
				{#if gameResult}
					<div class="game-over">
						<p class="result" class:won={gameResult.winner === PLAYER_IDX}>
							{gameMessage}
						</p>
						<button onclick={() => startNewGame()}>Play Again</button>
					</div>
				{:else}
					<p class="message">{gameMessage}</p>
					<div class="buttons">
						<button class="draw-btn" onclick={() => playerAction(1)} disabled={!playerTurn}>
							Draw
						</button>
						<button class="stay-btn" onclick={() => playerAction(0)} disabled={!playerTurn}>
							Stay
						</button>
					</div>
				{/if}
			</div>

			<div class="log">
				<h3>Game Log</h3>
				<div class="log-entries">
					{#each roundLog as entry}
						<div class="log-entry" class:round-header={entry.startsWith('---')}>{entry}</div>
					{/each}
				</div>
			</div>
		</div>
	{/if}
</main>

<style>
	:global(body) {
		margin: 0;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
		background: #1a1a2e;
		color: #e0e0e0;
	}

	main {
		max-width: 600px;
		margin: 0 auto;
		padding: 1rem;
	}

	h1 {
		text-align: center;
		font-size: 2rem;
		margin-bottom: 0.25rem;
		color: #e94560;
	}

	.subtitle {
		text-align: center;
		color: #888;
		margin-top: 0;
		margin-bottom: 1.5rem;
	}

	.start-screen {
		text-align: center;
		padding: 2rem;
	}

	button {
		padding: 0.75rem 1.5rem;
		font-size: 1rem;
		border: none;
		border-radius: 8px;
		cursor: pointer;
		font-weight: 600;
		transition: all 0.2s;
	}

	button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.scores {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1.5rem;
	}

	.score-card {
		background: #16213e;
		padding: 0.75rem 1.5rem;
		border-radius: 12px;
		text-align: center;
		min-width: 80px;
	}

	.score-card.winning {
		background: #0f3460;
		border: 2px solid #e94560;
	}

	.score-card .label {
		display: block;
		font-size: 0.8rem;
		color: #888;
		text-transform: uppercase;
	}

	.score-card .value {
		display: block;
		font-size: 1.5rem;
		font-weight: 700;
	}

	.round-info {
		color: #888;
		font-size: 0.9rem;
	}

	.hand-section {
		background: #16213e;
		border-radius: 12px;
		padding: 1rem;
		margin-bottom: 1rem;
	}

	.hand-section.player {
		border: 2px solid #0f3460;
	}

	.hand-section h2 {
		font-size: 1rem;
		margin: 0 0 0.75rem 0;
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.badge {
		font-size: 0.7rem;
		padding: 0.2rem 0.5rem;
		border-radius: 4px;
		font-weight: 600;
	}

	.badge.stayed { background: #2d6a4f; color: #b7e4c7; }
	.badge.busted { background: #9d0208; color: #ffccd5; }
	.badge.thinking { background: #0f3460; color: #a8dadc; }

	.cards {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
		min-height: 60px;
	}

	.card {
		width: 44px;
		height: 60px;
		background: #0f3460;
		border-radius: 8px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 1.25rem;
		font-weight: 700;
		border: 2px solid #533483;
	}

	.card.player-card { border-color: #e94560; }
	.card.busted { opacity: 0.4; border-color: #9d0208; }

	.hand-info {
		margin-top: 0.5rem;
		font-size: 0.85rem;
		color: #888;
	}

	.actions {
		text-align: center;
		margin: 1.5rem 0;
	}

	.message { margin-bottom: 0.75rem; color: #a8dadc; }

	.buttons {
		display: flex;
		gap: 1rem;
		justify-content: center;
	}

	.draw-btn { background: #e94560; color: white; }
	.draw-btn:hover:not(:disabled) { background: #c73e54; }
	.stay-btn { background: #2d6a4f; color: white; }
	.stay-btn:hover:not(:disabled) { background: #245a42; }

	.game-over { padding: 1rem; }
	.result { font-size: 1.25rem; font-weight: 700; color: #e94560; }
	.result.won { color: #52b788; }
	.game-over button { background: #e94560; color: white; margin-top: 0.75rem; }

	.log {
		background: #16213e;
		border-radius: 12px;
		padding: 1rem;
		margin-top: 1rem;
	}

	.log h3 { font-size: 0.9rem; margin: 0 0 0.5rem 0; color: #888; }

	.log-entries {
		max-height: 200px;
		overflow-y: auto;
		font-size: 0.8rem;
		font-family: monospace;
	}

	.log-entry { padding: 0.15rem 0; color: #aaa; }
	.log-entry.round-header { color: #e94560; font-weight: 600; margin-top: 0.25rem; }
</style>
