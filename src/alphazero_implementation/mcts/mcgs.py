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

    def __init__(self, game_state: State, N: int = 0, Q: float = 0):
        self.N = N
        self.Q = Q
        self.game_state = game_state
        self.children_and_edge_visits: dict[Action, tuple[Node, int]] = {}
        self.U: Value = [0.0] * self.game_state.config.num_players
        self.action_policy: ActionPolicy = {}

    @property
    def children(self) -> list["Node"]:
        return [child for (child, _) in self.children_and_edge_visits.values()]


def select_action_according_to_puct(node: Node, c_puct: float = 1.0) -> Action:
    """
    Select the action according to the PUCT formula.

    Args:
        node: The current node object, containing N(n), Q(n, a), N(n, a), and P(n, a) for each action 'a'.
        c_puct: The exploration constant for PUCT, controlling the balance between exploration and exploitation.

    Returns:
        The action 'a' that maximizes the PUCT formula.
    """

    # Initialize variables to store the best action and its corresponding value
    best_action: Action | None = None
    best_value = -float("inf")

    # Total visit count for all actions from the current node
    total_visits = node.N

    # action_policy, value = nn.predict([node.game_state])[0]

    # Loop through each action and compute its PUCT value
    for action in node.game_state.actions:
        # Compute Q value as average utility for this action
        child_node, edge_visits = node.children_and_edge_visits.get(action, (None, 0))
        Q_value = child_node.Q if child_node else 0.0

        # Prior probability for this action
        P_value = node.action_policy.get(
            action, 1.0 / len(node.game_state.actions)
        )  # Default to uniform if missing

        # PUCT value: PlayerToMove(n) * Q(n, a) + c_puct * P(n, a) * sqrt(total_visits) / (1 + N(n, a))
        puct_value = Q_value + c_puct * P_value * math.sqrt(total_visits) / (
            1 + edge_visits
        )

        # Update best action if the PUCT value is higher than the current best
        if puct_value > best_value:
            best_value = puct_value
            best_action = action

    return best_action  # type: ignore[no-any-return]
