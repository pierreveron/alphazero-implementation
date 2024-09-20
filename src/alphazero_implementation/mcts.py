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
    def __init__(self, exploration_weight: float = 1.4):
        self.exploration_weight: float = exploration_weight

    def choose(self, node: Node) -> Node:
        if not node.children:
            return self.expand(node)
        return max(node.children, key=self.uct_score)

    def expand(self, node: Node) -> Node:
        tried_children = {child.state for child in node.children}
        new_state = node.state.find_random_child()
        while new_state in tried_children:
            new_state = node.state.find_random_child()
        child = Node(new_state, parent=node)
        node.children.append(child)
        return child

    def simulate(self, node: Node) -> float:
        inplay = node.state
        while not inplay.is_terminal():
            inplay = inplay.find_random_child()
        return inplay.reward()

    def backpropagate(self, node: Node | None, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def uct_score(self, child: Node) -> float:
        parent_visits = child.parent.visits if child.parent else 0
        if child.visits == 0:
            return float("inf")
        return (child.value / child.visits) + self.exploration_weight * math.sqrt(
            math.log(parent_visits) / child.visits
        )

    def search(self, initial_state: GameState, n_simulations: int) -> Node:
        root = Node(initial_state)
        for _ in range(n_simulations):
            leaf = self.choose(root)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
        return max(root.children, key=lambda c: c.visits)
