import math
from typing import List, Optional, Protocol


class GameState(Protocol):
    def find_random_child(self) -> "GameState": ...

    def is_terminal(self) -> bool: ...

    def reward(self) -> float: ...


class Node:
    def __init__(self, state: GameState, parent: Optional["Node"] = None):
        self.state: GameState = state
        self.parent: Optional["Node"] = parent
        self.children: List["Node"] = []
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

    def __init__(self, exploration_weight: float = 1.4):
        """
        Initialize the MCTS object.

        Args:
            exploration_weight (float): The exploration weight for the UCT formula.
        """
        self.exploration_weight: float = exploration_weight

    def select(self, node: Node) -> Node:
        """
        Step 1: Selection - Select a child node using the UCT score.

        If the node has no children, expand it. Otherwise, choose the child
        with the highest UCT score.

        Args:
            node (Node): The current node.

        Returns:
            Node: The selected child node.
        """
        if not node.children:
            return self.expand(node)
        return max(node.children, key=self._uct_score)

    def expand(self, node: Node) -> Node:
        """
        Step 2: Expansion - Expand the given node by adding a new child.

        Creates a new child node with a random untried state.

        Args:
            node (Node): The node to expand.

        Returns:
            Node: The newly created child node.
        """
        tried_children = {child.state for child in node.children}
        new_state = node.state.find_random_child()
        while new_state in tried_children:
            new_state = node.state.find_random_child()
        child = Node(new_state, parent=node)
        node.children.append(child)
        return child

    def simulate(self, node: Node) -> float:
        """
        Step 3: Simulation - Simulate a random playout from the given node.

        Plays random moves until a terminal state is reached.

        Args:
            node (Node): The starting node for the simulation.

        Returns:
            float: The reward value of the terminal state.
        """
        inplay = node.state
        while not inplay.is_terminal():
            inplay = inplay.find_random_child()
        return inplay.reward()

    def backpropagate(self, node: Node | None, reward: float) -> None:
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

    def _uct_score(self, child: Node) -> float:
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

    def search(self, initial_state: GameState, n_simulations: int) -> Node:
        """
        Perform the MCTS search from the initial state.

        Runs the specified number of simulations and returns the best child of the root.

        Args:
            initial_state (GameState): The starting game state.
            n_simulations (int): The number of simulations to run.

        Returns:
            Node: The best child node of the root (best next move).
        """
        root = Node(initial_state)
        for _ in range(n_simulations):
            leaf = self.select(root)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
        return max(root.children, key=lambda c: c.visits)
