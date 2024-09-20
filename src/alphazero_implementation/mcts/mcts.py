import math

from alphazero_implementation.games.state import GameState
from alphazero_implementation.models.model import Model
from alphazero_implementation.models.random_model import RandomModel


class TreeNode:
    def __init__(self, state: GameState, parent: "TreeNode | None" = None):
        self.state: GameState = state
        self.parent: TreeNode | None = parent
        self.children: list[TreeNode] = []
        self.visits: int = 0
        self.value: float = 0


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation.

    This class implements the MCTS algorithm with the following steps:
    1. Selection: Choose a path through the tree to a leaf node using UCT.
    2. Expansion: Add a new child node to the selected leaf node.
    3. Simulation: Perform a random playout from the new node to a terminal state.
    4. Backpropagation: Update the statistics of all nodes in the selected path.

    The search process repeats these steps for a specified number of simulations,
    gradually building a tree of possible game states and their estimated values.
    """

    def __init__(
        self,
        *,
        exploration_weight: float = 1.4,
        simulation_model: Model = RandomModel(),
    ):
        """
        Initialize the MCTS object.

        Args:
            exploration_weight (float): The exploration weight for the UCT formula.
        """
        self.exploration_weight: float = exploration_weight
        self.simulation_model: Model = simulation_model

    def select(self, node: TreeNode) -> TreeNode:
        """
        Step 1: Selection - Select a child node using the UCT score.

        If the node has no children, attempt to expand it. If expansion is not possible
        (all actions explored), return the node itself. Otherwise, choose the child
        with the highest UCT score.

        Args:
            node (Node): The current node.

        Returns:
            Node: The selected child node or the input node if expansion is not possible.
        """
        while not node.state.is_terminal():
            if not node.children:
                try:
                    return self.expand(node)
                except ValueError:
                    # All actions have been explored, return this node
                    return node
            node = max(node.children, key=self._uct_score)
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        """
        Step 2: Expansion - Expand the given node by adding a new child.

        Creates a new child node for an unexplored action.

        Args:
            node (Node): The node to expand.

        Returns:
            Node: The newly created child node.
        """
        unexplored_actions = set(node.state.legal_actions) - {
            child.state.last_action for child in node.children
        }
        if not unexplored_actions:
            raise ValueError("All actions have been explored")

        action = unexplored_actions.pop()
        new_state = node.state.play(action)
        child = TreeNode(new_state, parent=node)
        node.children.append(child)
        return child

    def simulate(self, node: TreeNode) -> float:
        """
        Step 3: Simulation - Simulate a random playout from the given node.

        Plays random moves until a terminal state is reached.

        Args:
            node (Node): The starting node for the simulation.

        Returns:
            float: The reward value of the terminal state.
        """
        current_state = node.state
        while not current_state.is_terminal():
            action, _ = self.simulation_model.predict(current_state)
            current_state = current_state.play(action)
        return current_state.reward()

    def backpropagate(self, node: TreeNode | None, reward: float) -> None:
        """
        Step 4: Backpropagation - Backpropagate the reward through the tree.

        Updates the visit count and value for each node in the path.

        Args:
            node (Node | None): The leaf node to start backpropagation from.
            reward (float): The reward value to backpropagate.
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _uct_score(self, child: TreeNode) -> float:
        """
        Calculate the UCT score for a child node.

        Combines exploitation (node value) and exploration (visit count) terms.

        Args:
            child (Node): The child node to calculate the score for.

        Returns:
            float: The UCT score of the child node.
        """
        parent_visits = child.parent.visits if child.parent else 0
        if child.visits == 0:
            return float("inf")
        return (child.value / child.visits) + self.exploration_weight * math.sqrt(
            math.log(parent_visits) / child.visits
        )

    def search(self, initial_state: GameState, n_simulations: int) -> TreeNode:
        """
        Perform the MCTS search from the initial state.

        Runs the specified number of simulations and returns the best child of the root.

        Args:
            initial_state (State): The starting game state.
            n_simulations (int): The number of simulations to run.

        Returns:
            Node: The best child node of the root (best next move).
        """
        root = TreeNode(initial_state)
        for _ in range(n_simulations):
            leaf = self.select(root)
            if not leaf.state.is_terminal():
                reward = self.simulate(leaf)
                self.backpropagate(leaf, reward)
        return max(root.children, key=lambda c: c.visits) if root.children else root
