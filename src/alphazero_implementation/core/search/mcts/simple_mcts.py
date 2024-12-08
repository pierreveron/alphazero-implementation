import math
from typing import Generator

from simulator.game.connect import Action, State  # type: ignore[attr-defined]

from alphazero_implementation.core.training.episode import Episode, Sample
from alphazero_implementation.models.base import Model

from .node import Node


class SimpleMCTS:
    """A simple Monte Carlo Tree Search implementation for learning purposes.

    Note: This implementation makes individual model predictions for each node expansion.
    For efficiency in practice, predictions should be batched/parallelized like in AlphaZeroMCTS.
    """

    def __init__(
        self,
        *,
        model: Model,
        exploration_weight: float = 1.0,
    ):
        self.inference_model = model.get_inference_clone()
        self.exploration_weight = exploration_weight

    def update_inference_model(self, model: Model):
        """Update the inference model with the latest weights from the training model"""
        self.inference_model.load_state_dict(model.state_dict())
        self.inference_model.eval()

    def select_child(self, node: Node) -> Node:
        """Select a child node with the highest PUCT value.

        Calculate PUCT score as used in AlphaZero
        PUCT = Q + U where:
        U = c_puct * P * sqrt(N_parent) / (1 + N_child)
        Q = mean value of child node
        P = prior probability
        """
        best_score = float("-inf")
        best_child = None

        for child in node.children.values():
            q_value = child.value
            u = (
                self.exploration_weight
                * child.prior
                * math.sqrt(node.visit_count)
                / (1 + child.visit_count)
            )
            score = q_value + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child  # type: ignore[return-value]

    def expand(self, node: Node):
        """Expand a node by adding all possible children."""
        [policy], [values] = self.inference_model.predict([node.state])
        for action, prob in policy.items():
            next_state = action.sample_next_state()
            node.add_child(action, next_state, prob)

        return values[node.state.player]

    def backpropagate(self, node: Node | None, value: float):
        """Backpropagate the result of a simulation up the tree."""
        while node is not None:
            node.value_sum += value
            node.visit_count += 1

            value = -value  # Switch perspective between players
            node = node.parent

    def run(self, root: Node, num_simulations: int) -> dict[Action, float]:
        """Run the MCTS algorithm for a given number of simulations."""
        for _ in range(num_simulations):
            node = root

            # Selection
            while node.is_expanded:
                node = self.select_child(node)

            # Expansion and evaluation
            value: float = 0
            if node.is_terminal:
                value = node.utility_values[node.parent.state.player]  # type: ignore[attr-defined]
            else:
                value = self.expand(node)

            # Backpropagation
            self.backpropagate(node, value)

        return root.improved_policy

    def generate_episodes(
        self, initial_state: State, num_episodes: int, num_simulations: int
    ) -> Generator[Episode, None, None]:
        """Run the MCTS algorithm for a given number of simulations."""

        episode_count = 0
        while episode_count < num_episodes:
            current_node = Node(initial_state)
            episode = Episode()
            episode_count += 1
            while not current_node.is_terminal:
                for _ in range(num_simulations):
                    node = current_node

                    # Selection
                    while node.is_expanded:
                        node = self.select_child(node)

                    # Expansion and evaluation
                    value: float = 0
                    if node.is_terminal:
                        value = node.utility_values[node.parent.state.player]  # type: ignore[attr-defined]
                    else:
                        value = self.expand(node)

                    # Backpropagation
                    self.backpropagate(node, value)

                episode.add_sample(
                    Sample(
                        state=current_node.state,
                        policy=current_node.improved_policy,
                        value=[0.0] * current_node.state.config.num_players,
                    )
                )

                current_node = current_node.select_next_node()

            # The game ended
            episode.backpropagate_outcome(current_node.utility_values)

            yield episode
