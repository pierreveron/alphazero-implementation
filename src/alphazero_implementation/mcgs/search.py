import math
from typing import Generator

from simulator.game.connect import Action, State  # type: ignore[attr-defined]

from alphazero_implementation.alphazero.types import Episode, Sample
from alphazero_implementation.mcgs.node import Node
from alphazero_implementation.models.model import Model


class AlphaZeroMCTS:
    def __init__(
        self,
        *,
        model: Model,
        num_simulations: int,
        num_episodes: int,
        game_initial_state: State,
        exploration_weight: float = 1.0,
    ):
        self.inference_model = model.get_inference_clone()
        self.num_simulations = num_simulations
        self.num_episodes = num_episodes
        self.game_initial_state = game_initial_state
        self.exploration_weight = exploration_weight
        self.num_players = game_initial_state.config.num_players

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

    def run(self, root: Node) -> dict[Action, float]:
        """Run the MCTS algorithm for a given number of simulations."""
        current_nodes = [root]
        self.run_simulations(current_nodes)

        return root.improved_policy

    def generate_episodes_in_parallel(
        self, initial_state: State | None = None
    ) -> Generator[Episode, None, None]:
        """Run the MCTS algorithm for a given number of simulations."""
        if initial_state is None:
            initial_state = self.game_initial_state

        episodes = [Episode() for _ in range(self.num_episodes)]
        current_nodes: list[Node] = [
            Node(initial_state) for _ in range(self.num_episodes)
        ]

        episode_count = 0
        while True:
            # Run all simulations first
            self.run_simulations(current_nodes)

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

                current_nodes[current_node_index] = Node(initial_state)
                episodes[current_node_index] = Episode()
                episode_count += 1

                if episode_count >= self.num_episodes:
                    return

    def run_simulations(self, current_nodes: list[Node]) -> None:
        for _ in range(self.num_simulations):
            # Track nodes that need expansion
            nodes_to_expand: list[Node] = []

            for current_node in current_nodes:
                node = current_node

                while node.is_expanded:
                    node = self.select_child(node)

                if node.is_terminal:
                    value = node.state.reward.tolist()[node.parent.state.player]  # type: ignore[attr-defined]
                    self.backpropagate(node, value)
                else:
                    nodes_to_expand.append(node)

            # Batch prediction for all nodes that need expansion
            if nodes_to_expand:
                states = [node.state for node in nodes_to_expand]
                policies, values = self.inference_model.predict(states)

                # Expand all nodes with their predictions
                for node, policy, value in zip(nodes_to_expand, policies, values):
                    for action, prob in policy.items():
                        next_state = action.sample_next_state()
                        node.add_child(action, next_state, prob)
                    self.backpropagate(node, value[node.state.player])
