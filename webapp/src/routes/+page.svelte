<script lang="ts">
	import { onMount } from "svelte";
	import {
		Game,
		handSum,
		isDone,
		roundScore,
		WINNING_SCORE,
		type GameResult,
	} from "$lib/gameEngine";
	import { OnnxAgent } from "$lib/onnxAgent";
	import { QAgent } from "$lib/qAgent";

	type AgentType = "random" | "q" | "ppo";
	type AgentStats = { wins: number; losses: number };
	type Stats = Record<AgentType, AgentStats>;

	const PLAYER_IDX = 0;
	const AI_IDX = 1;

	function loadStats(): Stats {
		const defaults: Stats = {
			random: { wins: 0, losses: 0 },
			q: { wins: 0, losses: 0 },
			ppo: { wins: 0, losses: 0 },
		};
		if (typeof document === "undefined") return defaults;
		const match = document.cookie.match(/flip7_stats=([^;]+)/);
		if (!match) return defaults;
		try {
			return { ...defaults, ...JSON.parse(decodeURIComponent(match[1])) };
		} catch {
			return defaults;
		}
	}

	function saveStats(s: Stats) {
		const expires = new Date();
		expires.setFullYear(expires.getFullYear() + 1);
		document.cookie = `flip7_stats=${encodeURIComponent(JSON.stringify(s))}; expires=${expires.toUTCString()}; path=/`;
	}

	function winRate(s: AgentStats): string {
		const total = s.wins + s.losses;
		if (total === 0) return "–";
		return `${Math.round((s.wins / total) * 100)}%`;
	}

	let stats = $state(loadStats());
	let game: Game | null = $state(null);
	let onnxAgent = new OnnxAgent();
	let qAgent = new QAgent();
	let onnxReady = $state(false);
	let qReady = $state(false);
	let selectedAgent: AgentType | null = $state(null);
	let playerTurn = $state(false);
	let gameMessage = $state("Select an opponent.");
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
	let deckCounts = $state<[number, number][]>([]);
	let deckRemaining = $state(0);
	let ezMode = $state(false);

	let bustProb = $derived.by(() => {
		if (!game || game.gameOver || deckRemaining === 0) return 0;
		let bustCount = 0;
		for (const [val, cnt] of deckCounts) {
			if (playerHand.includes(val)) bustCount += cnt;
		}
		return bustCount / deckRemaining;
	});

	let evDraw = $derived.by(() => {
		if (!game || game.gameOver || deckRemaining === 0) return 0;
		let ev = 0;
		for (const [val, cnt] of deckCounts) {
			if (!playerHand.includes(val)) {
				ev += (cnt / deckRemaining) * (playerHandSum + val);
			}
		}
		return ev;
	});

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
		deckCounts = [...game.deck.cardCounts().entries()].sort(
			(a, b) => a[0] - b[0],
		);
		deckRemaining = game.deck.remaining();
	}

	onMount(async () => {
		stats = loadStats();
		const results = await Promise.allSettled([
			qAgent.init("/q_table.json").then(() => {
				qReady = true;
			}),
			onnxAgent.init("/ppo_flip7.onnx", "/normalize_stats.json").then(() => {
				onnxReady = true;
			}),
		]);
		gameMessage = "Select an opponent.";
	});

	function startNewGame(agentType: AgentType) {
		selectedAgent = agentType;
		const seed = Math.floor(Math.random() * 2147483647);
		game = new Game(seed);
		gameResult = null;
		syncState();
		startNewRound();
	}

	function startNewRound() {
		if (!game || game.gameOver) return;

		game.currentPlayerIdx = game.roundNumber % 2;
		game.startRound();
		syncState();

		if (game.currentPlayerIdx === AI_IDX) {
			playerTurn = false;
			runAiTurn();
		} else {
			playerTurn = true;
			gameMessage = "Your turn! Draw or Stay?";
		}
	}

	async function runAiTurn() {
		if (!game || game.gameOver) return;
		aiThinking = true;

		while (game.currentPlayerIdx === AI_IDX && !game.isRoundOver()) {
			await sleep(500);

			let action: 0 | 1;
			if (selectedAgent === "random") {
				action = Math.random() < 0.5 ? 1 : 0;
			} else if (selectedAgent === "ppo" && onnxReady) {
				action = await onnxAgent.chooseAction(
					game.players[AI_IDX],
					game.players[PLAYER_IDX],
					game,
				);
			} else if (selectedAgent === "q" && qReady) {
				action = qAgent.chooseAction(game.players[AI_IDX]);
			} else {
				action = Math.random() < 0.4 ? 1 : 0;
			}

			const roundOver = game.step(action);
			syncState();

			if (roundOver) {
				endRound();
				return;
			}
		}

		aiThinking = false;

		if (!game.isRoundOver() && !isDone(game.players[PLAYER_IDX])) {
			playerTurn = true;
			gameMessage = "Your turn! Draw or Stay?";
		}
	}

	async function playerAction(action: 0 | 1) {
		if (!game || !playerTurn || game.gameOver) return;
		playerTurn = false;

		const roundOver = game.step(action);
		syncState();

		if (roundOver) {
			endRound();
			return;
		}

		if (game.currentPlayerIdx === AI_IDX && !isDone(game.players[AI_IDX])) {
			await runAiTurn();
		} else if (!isDone(game.players[PLAYER_IDX])) {
			playerTurn = true;
			gameMessage = "Your turn! Draw or Stay?";
		}
	}

	function endRound() {
		if (!game) return;
		aiThinking = false;
		syncState();

		if (game.gameOver) {
			gameResult = game.result;
			if (gameResult!.winner === PLAYER_IDX) {
				gameMessage = `You win! Final: ${gameResult!.scores[0]} - ${gameResult!.scores[1]}`;
				if (selectedAgent) {
					stats[selectedAgent].wins++;
					saveStats(stats);
				}
			} else {
				gameMessage = `AI wins! Final: ${gameResult!.scores[0]} - ${gameResult!.scores[1]}`;
				if (selectedAgent) {
					stats[selectedAgent].losses++;
					saveStats(stats);
				}
			}
		} else {
			gameMessage = "Round over! Starting next round...";
			setTimeout(() => startNewRound(), 1500);
		}
	}

	function sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}
</script>

<svelte:head>
	<title>Flip 7 - Play Against AI</title>
</svelte:head>

<div class="corner-menu">
	<div class="menu-pill">
		Rules
		<div class="tooltip rules-tooltip">
			<strong>How to Play</strong>
			<ul>
				<li>
					<b>Deck:</b> Cards 1–12; value <i>v</i> appears <i>v</i> times (78 total)
				</li>
				<li>
					<b>Draw:</b> Flip a card. Already have it? Bust — score 0 this round
				</li>
				<li><b>Stay:</b> Lock in your hand sum for the round</li>
				<li><b>Win:</b> First to reach 200 total points</li>
				<li>
					<b>Tiebreaker:</b> Both cross 200 same round → highest score wins
				</li>
				<li><b>Deck:</b> Persists across rounds; reshuffles only when empty</li>
			</ul>
		</div>
	</div>
	<div class="menu-pill">
		Stats
		<div class="tooltip stats-tooltip">
			<strong>Your Win Rate</strong>
			<div class="stat-row">
				<span class="stat-label">Easy (Random)</span>
				<span
					>{stats.random.wins}W – {stats.random.losses}L ({winRate(
						stats.random,
					)})</span
				>
			</div>
			<div class="stat-row">
				<span class="stat-label">Medium (Q Learning)</span>
				<span>{stats.q.wins}W – {stats.q.losses}L ({winRate(stats.q)})</span>
			</div>
			<div class="stat-row">
				<span class="stat-label">Hard (PPO)</span>
				<span
					>{stats.ppo.wins}W – {stats.ppo.losses}L ({winRate(stats.ppo)})</span
				>
			</div>
		</div>
	</div>
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<div
		class="menu-pill ez-toggle"
		role="button"
		tabindex="0"
		onclick={() => (ezMode = !ezMode)}
	>
		{ezMode ? "Hide E(Draw) and P(Bust)" : "Show E(Draw) and P(Bust)"}
	</div>
</div>

<main>
	<h1>Flip 7</h1>
	<p class="subtitle">A push your luck card game</p>

	{#if !game}
		<div class="start-screen">
			<p>{gameMessage}</p>
			<div class="difficulty-buttons">
				<button
					class="difficulty-btn easy"
					onclick={() => startNewGame("random")}
				>
					Easy (Random)
				</button>
				<button
					class="difficulty-btn medium"
					onclick={() => startNewGame("q")}
					disabled={!qReady}
				>
					Medium (Q Learning)
				</button>
				<button
					class="difficulty-btn hard"
					onclick={() => startNewGame("ppo")}
					disabled={!onnxReady}
				>
					Hard (PPO)
				</button>
			</div>
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
						<div class="card player-card" class:busted={playerBusted}>
							{card}
						</div>
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
						<p class="play-again">Play again? Select an opponent.</p>
						<div class="difficulty-buttons">
							<button
								class="difficulty-btn easy"
								onclick={() => startNewGame("random")}
							>
								Easy (Random)
							</button>
							<button
								class="difficulty-btn medium"
								onclick={() => startNewGame("q")}
								disabled={!qReady}
							>
								Medium (Q Learning)
							</button>
							<button
								class="difficulty-btn hard"
								onclick={() => startNewGame("ppo")}
								disabled={!onnxReady}
							>
								Hard (PPO)
							</button>
						</div>
					</div>
				{:else}
					<p class="message">{gameMessage}</p>
					<div class="buttons">
						<button
							class="draw-btn"
							onclick={() => playerAction(1)}
							disabled={!playerTurn}
						>
							Draw
						</button>
						<button
							class="stay-btn"
							onclick={() => playerAction(0)}
							disabled={!playerTurn}
						>
							Stay
						</button>
					</div>
					{#if ezMode}
						<div class="ez-stats">
							<span class="ez-stat">
								<span class="ez-label">P(Bust)</span>
								<span class="ez-value" class:danger={bustProb > 0.5}
									>{(bustProb * 100).toFixed(1)}%</span
								>
							</span>
							<span class="ez-divider">·</span>
							<span class="ez-stat">
								<span class="ez-label">E(Draw)</span>
								<span class="ez-value">{evDraw.toFixed(1)}</span>
							</span>
						</div>
					{/if}
				{/if}
			</div>

			<div class="deck-section">
				<h3>
					Deck <span class="deck-remaining"
						>({deckRemaining} cards remaining)</span
					>
				</h3>
				<div class="deck-grid">
					{#each deckCounts as [value, count]}
						<div class="deck-cell" class:empty={count === 0}>
							<span class="deck-value">{value}</span>
							<span class="deck-count">{count}/{value}</span>
						</div>
					{/each}
				</div>
			</div>
		</div>
	{/if}
</main>

<style>
	:global(body) {
		margin: 0;
		font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
			sans-serif;
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

	.badge.stayed {
		background: #2d6a4f;
		color: #b7e4c7;
	}
	.badge.busted {
		background: #9d0208;
		color: #ffccd5;
	}
	.badge.thinking {
		background: #0f3460;
		color: #a8dadc;
	}

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

	.card.player-card {
		border-color: #e94560;
	}
	.card.busted {
		opacity: 0.4;
		border-color: #9d0208;
	}

	.hand-info {
		margin-top: 0.5rem;
		font-size: 0.85rem;
		color: #888;
	}

	.actions {
		text-align: center;
		margin: 1.5rem 0;
	}

	.message {
		margin-bottom: 0.75rem;
		color: #a8dadc;
	}

	.buttons {
		display: flex;
		gap: 1rem;
		justify-content: center;
	}

	.draw-btn {
		background: #e94560;
		color: white;
	}
	.draw-btn:hover:not(:disabled) {
		background: #c73e54;
	}
	.stay-btn {
		background: #2d6a4f;
		color: white;
	}
	.stay-btn:hover:not(:disabled) {
		background: #245a42;
	}

	.game-over {
		padding: 1rem;
	}
	.result {
		font-size: 1.25rem;
		font-weight: 700;
		color: #e94560;
	}
	.result.won {
		color: #52b788;
	}
	.play-again {
		color: #a8dadc;
		margin-bottom: 0;
	}

	.difficulty-buttons {
		display: flex;
		gap: 1rem;
		justify-content: center;
		margin-top: 1rem;
	}

	.difficulty-btn {
		padding: 0.75rem 1.5rem;
		color: white;
		white-space: nowrap;
	}

	.difficulty-btn.easy {
		background: #2d6a4f;
	}
	.difficulty-btn.easy:hover:not(:disabled) {
		background: #245a42;
	}
	.difficulty-btn.medium {
		background: #c89b3c;
		color: #1a1a2e;
	}
	.difficulty-btn.medium:hover:not(:disabled) {
		background: #a8832e;
	}
	.difficulty-btn.hard {
		background: #e94560;
	}
	.difficulty-btn.hard:hover:not(:disabled) {
		background: #c73e54;
	}

	.deck-section {
		background: #16213e;
		border-radius: 12px;
		padding: 1rem;
		margin-top: 1rem;
	}

	.deck-section h3 {
		font-size: 0.9rem;
		margin: 0 0 0.75rem 0;
		color: #888;
	}

	.deck-remaining {
		font-weight: 400;
	}

	.deck-grid {
		display: grid;
		grid-template-columns: repeat(6, 1fr);
		gap: 0.4rem;
	}

	.deck-cell {
		background: #0f3460;
		border-radius: 6px;
		padding: 0.4rem 0.25rem;
		text-align: center;
		border: 1px solid #533483;
	}

	.deck-cell.empty {
		opacity: 0.25;
	}

	.deck-value {
		display: block;
		font-size: 1rem;
		font-weight: 700;
	}

	.deck-count {
		display: block;
		font-size: 0.65rem;
		color: #888;
		margin-top: 0.1rem;
	}

	/* Corner menu */
	.corner-menu {
		position: fixed;
		top: 1rem;
		right: 1rem;
		display: flex;
		gap: 0.5rem;
		z-index: 100;
	}

	.menu-pill {
		position: relative;
		background: #16213e;
		border: 1px solid #533483;
		border-radius: 20px;
		padding: 0.35rem 0.85rem;
		font-size: 0.8rem;
		color: #a8dadc;
		cursor: default;
		user-select: none;
		white-space: nowrap;
	}

	.menu-pill:hover {
		border-color: #a8dadc;
	}

	.ez-toggle {
		cursor: pointer;
	}

	.tooltip {
		display: none;
		position: absolute;
		top: calc(100% + 0.5rem);
		right: 0;
		background: #0f1b38;
		border: 1px solid #533483;
		border-radius: 10px;
		padding: 0.75rem 1rem;
		min-width: 240px;
		font-size: 0.8rem;
		color: #e0e0e0;
		line-height: 1.5;
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
	}

	.menu-pill:hover .tooltip {
		display: block;
	}

	.rules-tooltip ul {
		margin: 0.5rem 0 0 0;
		padding-left: 1.1rem;
	}

	.rules-tooltip li {
		margin-bottom: 0.35rem;
	}

	.stats-tooltip strong {
		display: block;
		margin-bottom: 0.5rem;
		color: #a8dadc;
	}

	.stat-row {
		display: flex;
		justify-content: space-between;
		gap: 1rem;
		padding: 0.25rem 0;
		border-bottom: 1px solid #1e2f5a;
	}

	.stat-row:last-child {
		border-bottom: none;
	}

	.stat-label {
		color: #888;
	}

	.ez-stats {
		display: flex;
		align-items: center;
		justify-content: center;
		margin-top: 1rem;
		gap: 0.6rem;
		font-size: 0.85rem;
	}

	.ez-stat {
		display: flex;
		gap: 0.3rem;
		align-items: baseline;
	}

	.ez-label {
		color: #888;
		font-size: 0.75rem;
	}

	.ez-value {
		font-weight: 700;
		color: #a8dadc;
	}

	.ez-value.danger {
		color: #e94560;
	}

	.ez-divider {
		color: #533483;
	}
</style>
