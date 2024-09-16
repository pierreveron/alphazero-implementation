import math


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


class MCTS:
    def __init__(self, exploration_weight=1.4):
        self.exploration_weight = exploration_weight

    def choose(self, node):
        if not node.children:
            return self.expand(node)
        return max(node.children, key=self.uct_score)

    def expand(self, node):
        tried_children = {child.state for child in node.children}
        new_state = node.state.find_random_child()
        while new_state in tried_children:
            new_state = node.state.find_random_child()
        child = Node(new_state, parent=node)
        node.children.append(child)
        return child

    def simulate(self, node):
        inplay = node.state
        while not inplay.is_terminal():
            inplay = inplay.find_random_child()
        return inplay.reward()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def uct_score(self, child):
        parent_visits = child.parent.visits
        if child.visits == 0:
            return float("inf")
        return (child.value / child.visits) + self.exploration_weight * math.sqrt(
            math.log(parent_visits) / child.visits
        )

    def search(self, initial_state, n_simulations):
        root = Node(initial_state)
        for _ in range(n_simulations):
            leaf = self.choose(root)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
        return max(root.children, key=lambda c: c.visits)
