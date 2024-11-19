import math

from simulator.game.connect import Action, State  # type: ignore[import]

from alphazero_implementation.models.model import ActionPolicy, Value


class Node:
    """
    Represents a node in the Monte Carlo Graph Search (MCGS) graph.

    This class encapsulates the state and statistics of a single node in the MCTS graph.
    Each node corresponds to a game state and maintains information necessary for the
    MCGS algorithm, including visit counts, value estimates, and child nodes.

    Attributes:
        N (int): The number of times this node has been visited during MCGS.
        Q (float): The cumulative value of all simulations through this node.
        game_state (State): The game state represented by this node.
        children_and_edge_visits (dict): A dictionary mapping actions to tuples of
            (child Node, edge visit count). Represents the child nodes and the number
            of times each edge (action) has been traversed.
        U (NDArrayFloat): An array of utility values for each possible action
            from this state. Used for action selection and value estimation.

    The Node class is central to the MCTS algorithm, allowing the tree to be built
    and traversed efficiently while maintaining all necessary statistics for the
    Upper Confidence Bound (UCB) calculations and value backpropagation.
    """

    def __init__(self, game_state: State):
        self.game_state = game_state
        self.visit_count = 0
        self.cumulative_value = 0.0
        self.children_and_edge_visits: dict[Action, tuple[Node, int]] = {}
        self.utility_values: Value = [0.0] * self.game_state.config.num_players
        self.action_policy: ActionPolicy = {}

    @property
    def children(self) -> list["Node"]:
        return [child for (child, _) in self.children_and_edge_visits.values()]

    @property
    def is_terminal(self) -> bool:
        return self.game_state.has_ended

    def puct_score(self, action: Action, c_puct: float = 1.0) -> float:
        # Total visit count for all actions from the current node
        # total_visits = self.visit_count
        total_visits = sum(
            edge_visits for _, edge_visits in self.children_and_edge_visits.values()
        )

        child_node, edge_visits = self.children_and_edge_visits.get(action, (None, 0))
        Q_value = child_node.cumulative_value if child_node else 0.0

        # Prior probability for this action
        P_value = self.action_policy.get(
            action, 1.0 / len(self.game_state.actions)
        )  # Default to uniform if missing

        # PUCT value: PlayerToMove(n) * Q(n, a) + c_puct * P(n, a) * sqrt(total_visits) / (1 + N(n, a))
        puct_value = Q_value + c_puct * P_value * math.sqrt(total_visits) / (
            1 + edge_visits
        )

        return puct_value

    def select_action(self, c_puct: float = 1.0) -> Action:
        """Select the action with the highest PUCT score."""
        scores = {
            action: self.puct_score(action, c_puct)
            for action in self.game_state.actions
        }
        if not scores:
            raise ValueError("No valid actions available")
        return max(scores, key=scores.get)  # type: ignore[arg-type]
