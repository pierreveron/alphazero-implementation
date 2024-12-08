import math

from simulator.game.connect import Action  # type: ignore[attr-defined]

from alphazero_implementation.models.model import Model

from .node import Node


class AlphaZeroSearch:
    def __init__(
        self,
        *,
        model: Model,
        num_simulations: int,
        exploration_weight: float = 1.0,
    ):
        self.inference_model = model.get_inference_clone()
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight

    def update_inference_model(self, model: Model):
        """Update the inference model with the latest weights from the training model"""
        self.inference_model.load_state_dict(model.state_dict())
        self.inference_model.eval()

    def select_child(self, node: Node) -> Node:
        """Select a child node with the highest PUCT value."""
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

    def run_simulations(self, current_nodes: list[Node]) -> None:
        for _ in range(self.num_simulations):
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
