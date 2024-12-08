from typing import Generator

from simulator.game.connect import State  # type: ignore[attr-defined]

from alphazero_implementation.core.training.episode import Episode, Sample
from alphazero_implementation.models.model import Model
from alphazero_implementation.search.mcts.node import Node
from alphazero_implementation.search.mcts.search import AlphaZeroSearch


class EpisodeGenerator:
    def __init__(
        self,
        *,
        model: Model,
        num_simulations: int,
        num_episodes: int,
        game_initial_state: State,
        exploration_weight: float = 1.0,
    ):
        self.search = AlphaZeroSearch(
            model=model,
            num_simulations=num_simulations,
            exploration_weight=exploration_weight,
        )
        self.num_episodes = num_episodes
        self.game_initial_state = game_initial_state
        self.num_players = game_initial_state.config.num_players

    def update_inference_model(self, model: Model):
        """Update the inference model with the latest weights from the training model"""
        self.search.update_inference_model(model)

    def generate_episodes(
        self, initial_state: State | None = None
    ) -> Generator[Episode, None, None]:
        """Generate episodes using MCTS."""
        if initial_state is None:
            initial_state = self.game_initial_state

        episodes = [Episode() for _ in range(self.num_episodes)]
        current_nodes: list[Node] = [
            Node(initial_state) for _ in range(self.num_episodes)
        ]

        episode_count = 0
        while True:
            # Run all simulations first
            self.search.run_simulations(current_nodes)

            # After all simulations are complete, then collect samples
            for current_node_index, current_node in enumerate(current_nodes):
                episode = episodes[current_node_index]

                episode.add_sample(
                    Sample(
                        state=current_node.state,
                        policy=current_node.improved_policy,
                        value=[0.0] * self.num_players,
                    )
                )

                next_node = current_node.select_next_node()

                if not next_node.is_terminal:
                    current_nodes[current_node_index] = next_node
                    continue

                # The game ended so backpropagate the outcome to the previous game state
                outcome: list[float] = next_node.state.reward.tolist()  # type: ignore[attr-defined]
                episode.backpropagate_outcome(outcome)

                yield episode

                # Reset the current node for the next episode
                current_nodes[current_node_index] = Node(initial_state)
                episodes[current_node_index] = Episode()
                episode_count += 1

                if episode_count >= self.num_episodes:
                    return
