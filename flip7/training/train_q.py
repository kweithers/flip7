"""Q-learning training loop for Flip 7.

Simplified approach: train the Q-agent to maximize expected round score.
Each "episode" is a single round where the agent makes draw/stay decisions.
- STAY → reward = hand_sum / 78, episode ends
- DRAW + bust → reward = 0, episode ends (you score nothing)
- DRAW + safe → reward = 0, transition to new state

The key insight: with bust=0 and stay=hand_sum/78, the agent naturally
learns that drawing is +EV when bust probability is low and the expected
future hand sum exceeds the current sum.
"""

import random

from flip7.agents.q_agent import QAgent, hand_to_state
from flip7.agents.random_agent import RandomAgent
from flip7.engine.deck import Deck
from flip7.engine.game import Action, Game


def train_q_agent(
    num_episodes: int = 1_000_000,
    seed: int = 42,
    log_interval: int = 50_000,
) -> QAgent:
    rng = random.Random(seed)
    agent = QAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99999,
        rng=random.Random(rng.randint(0, 10**9)),
    )

    total_reward = 0.0
    total_stays = 0
    total_busts = 0
    total_cards_at_stay = 0

    for episode in range(num_episodes):
        deck = Deck(rng=random.Random(rng.randint(0, 10**9)))

        # Deal initial card
        hand = [deck.draw()]
        done = False

        while not done:
            state = hand_to_state(hand)
            action = agent.choose_action_from_state(state)

            if action == 0:  # STAY
                reward = sum(hand) / 78.0
                agent.update(state, action, reward, None, True)
                total_reward += reward
                total_stays += 1
                total_cards_at_stay += len(hand)
                done = True
            else:  # DRAW
                card = deck.draw()
                if card in set(hand):
                    # Bust — score 0 for the round
                    agent.update(state, action, 0.0, None, True)
                    total_busts += 1
                    done = True
                else:
                    hand.append(card)
                    next_state = hand_to_state(hand)
                    agent.update(state, action, 0.0, next_state, False)

        agent.decay_epsilon()

        if (episode + 1) % log_interval == 0:
            n = log_interval
            avg_reward = total_reward / n
            bust_rate = total_busts / n
            avg_cards = total_cards_at_stay / max(1, total_stays)
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"ε={agent.epsilon:.4f} | "
                f"Avg reward={avg_reward:.3f} | "
                f"Bust rate={bust_rate:.3f} | "
                f"Avg cards@stay={avg_cards:.1f} | "
                f"States={len(agent.q_table)}"
            )
            total_reward = 0.0
            total_stays = 0
            total_busts = 0
            total_cards_at_stay = 0

    return agent


def evaluate_vs_random(
    agent: QAgent, num_games: int = 10_000, seed: int = 123
) -> float:
    """Evaluate Q-agent win rate vs a random agent (draw_prob=0.5)."""
    rng = random.Random(seed)
    random_agent = RandomAgent(draw_prob=0.5, rng=random.Random(rng.randint(0, 10**9)))

    class GreedyWrapper:
        def choose_action(self, player, opponent, game):
            return agent.choose_action_greedy(player, opponent, game)

    greedy = GreedyWrapper()
    wins = 0

    for i in range(num_games):
        game = Game(seed=rng.randint(0, 10**9))
        if i % 2 == 0:
            agents = [greedy, random_agent]
            result = game.play_game(agents)
            if result.winner == 0:
                wins += 1
        else:
            agents = [random_agent, greedy]
            result = game.play_game(agents)
            if result.winner == 1:
                wins += 1

    return wins / num_games


if __name__ == "__main__":
    print("Training Q-learning agent...")
    agent = train_q_agent(num_episodes=1_000_000)
    agent.save("models/q_table.pkl")
    print(f"\nSaved Q-table with {len(agent.q_table)} states")

    print("\nEvaluating vs random agent (draw_prob=0.5)...")
    win_rate = evaluate_vs_random(agent)
    print(f"Win rate vs random (0.5): {win_rate:.3f}")

    # Also test vs more conservative random
    print("\nEvaluating vs random agent (draw_prob=0.3)...")
    rng = random.Random(456)
    conservative = RandomAgent(draw_prob=0.3, rng=random.Random(rng.randint(0, 10**9)))

    class GreedyWrapper:
        def choose_action(self, player, opponent, game):
            return agent.choose_action_greedy(player, opponent, game)

    greedy = GreedyWrapper()
    wins = 0
    num_games = 10_000
    for i in range(num_games):
        game = Game(seed=rng.randint(0, 10**9))
        if i % 2 == 0:
            agents_list = [greedy, conservative]
            result = game.play_game(agents_list)
            if result.winner == 0:
                wins += 1
        else:
            agents_list = [conservative, greedy]
            result = game.play_game(agents_list)
            if result.winner == 1:
                wins += 1
    print(f"Win rate vs random (0.3): {wins / num_games:.3f}")
