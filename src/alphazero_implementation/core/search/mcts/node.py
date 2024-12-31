from __future__ import annotations

import numpy as np
from simulator.game.connect import Action, State  # type: ignore[attr-defined]


class Node:
    def __init__(self, state: State, parent: Node | None = None, prior: float = 0.0):
        self.state = state  # The state of the game at this node
        self.parent = parent  # The parent node
        self.children: dict[
            Action, Node
        ] = {}  # Dictionary to store child nodes with actions as keys
        self.visit_count = 0  # Number of times this node has been visited
        self.value_sum = 0.0  # The sum of the values of this node
        self.prior = prior  # The prior probability of this node

    @property
    def raw_policy(self) -> dict[Action, float]:
        """The raw policy of this node."""
        return {action: child.prior for action, child in self.children.items()}

    @property
    def improved_policy(self) -> dict[Action, float]:
        """The improved policy of this node."""
        return {
            action: child.visit_count / (self.visit_count - 1)
            for action, child in self.children.items()
        }

    def select_next_node(self) -> Node:
        """Select the next node according to the improved policy."""
        index = np.random.choice(
            len(self.improved_policy), p=list(self.improved_policy.values())
        )
        action = list(self.improved_policy.keys())[index]
        new_node = Node(
            state=action.sample_next_state(),
            parent=self,
            prior=self.improved_policy[action],
        )
        return new_node

    def add_child(self, action: Action, child_state: State, prior: float):
        """Add a child node for a given action and state."""
        child_node = Node(state=child_state, parent=self, prior=prior)
        self.children[action] = child_node
        return child_node

    @property
    def value(self) -> float:
        """The value of this node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def is_terminal(self) -> bool:
        return self.state.has_ended

    @property
    def utility_values(self) -> list[float]:
        """The utility values of this node."""
        return self.state.reward.tolist()  # type: ignore[attr-defined]

    @property
    def is_root(self) -> bool:
        """Check if the node is the root node (no parent)."""
        return self.parent is None
