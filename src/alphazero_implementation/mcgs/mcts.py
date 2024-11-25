import math
from typing import Generator

from simulator.game.connect import Action, State

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
        self.model = model
        self.inference_model = self.model.get_inference_clone()
        self.num_simulations = num_simulations
        self.num_episodes = num_episodes
        self.game_initial_state = game_initial_state
        self.exploration_weight = exploration_weight

    def update_inference_model(self):
        """Update the inference model with the latest weights from the training model"""
        self.inference_model.load_state_dict(self.model.state_dict())
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

        return best_child  # type: ignore

    def expand(self, node: Node):
        """Expand a node by adding all possible children."""
        [policy], _ = self.model.predict([node.state])
        for action, prob in policy.items():
            next_state = action.sample_next_state()
            node.add_child(action, next_state, prob)

    def simulate(self, node: Node) -> float:
        """Use the neural network to predict the value of the state."""
        _, [value] = self.model.predict([node.state])
        return value[node.state.player]

    def backpropagate(self, node: Node | None, value: float):
        """Backpropagate the result of a simulation up the tree."""
        while node is not None:
            node.value_sum += value
            node.visit_count += 1

            value = -value  # Switch perspective between players
            node = node.parent

    def run(self, root: Node, num_simulations: int) -> dict[Action, float]:
        """Run the MCTS algorithm for a given number of simulations."""
        for idx in range(num_simulations):
            node = root

            # Selection
            while node.is_expanded:
                node = self.select_child(node)

            # Expansion and evaluation
            value: float = 0
            if node.is_terminal:
                value = node.state.reward.tolist()[node.parent.state.player]  # type: ignore
            else:
                self.expand(node)
                _, [values] = self.model.predict([node.state])
                value = values[node.state.player]

            # Backpropagation
            self.backpropagate(node, value)

        return root.improved_policy

    def generate_episodes(
        self, initial_state: State | None = None
    ) -> Generator[Episode, None, None]:
        """Run the MCTS algorithm for a given number of simulations."""
        if initial_state is None:
            initial_state = self.game_initial_state

        episode_count = 0
        while episode_count < self.num_episodes:
            current_node = Node(initial_state)
            episode = Episode()
            episode_count += 1
            while not current_node.is_terminal:
                for idx in range(self.num_simulations):
                    node = current_node

                    # Selection
                    while node.is_expanded:
                        node = self.select_child(node)

                    # Expansion and evaluation
                    value: float = 0
                    if node.is_terminal:
                        value = node.state.reward.tolist()[node.parent.state.player]  # type: ignore
                    else:
                        self.expand(node)
                        _, [values] = self.model.predict([node.state])
                        value = values[node.state.player]

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
            outcome = current_node.state.reward.tolist()  # type: ignore[attr-defined]

            # episode.add_sample(
            #     Sample(
            #         state=current_node.state,
            #         policy={},
            #         value=outcome,
            #     )
            # )

            # Determine game outcome (e.g., +1 for win, -1 for loss, 0 for draw)
            episode_history = episode.samples
            for i in range(len(episode_history)):
                episode_history[i].value = outcome

            yield episode
