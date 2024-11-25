import math

from alphazero_implementation.mcts_v2.node import Node
from alphazero_implementation.models.model import Model


class AlphaZeroMCTS:
    def __init__(self, model: Model, exploration_weight: float = 1.0):
        self.model = model
        self.exploration_weight = exploration_weight

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

    def run(self, root: Node, num_simulations: int):
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

        improved_policy = {
            action: child.visit_count / root.visit_count
            for action, child in root.children.items()
        }

        return improved_policy
