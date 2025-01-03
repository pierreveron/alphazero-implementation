from __future__ import annotations

import math

import numpy as np

from .base_game import BaseGame
from .base_model import BaseModel


def ucb_score(parent: Node, child: Node) -> float:
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior: float, to_play: int, parent: Node | None = None):
        self.visit_count: int = 0
        self.to_play: int = to_play
        self.prior: float = prior
        self.value_sum: float = 0.0
        self.children: dict[int, Node] = {}
        self.state: np.ndarray | None = None
        self.parent: Node | None = parent

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature: float) -> int:
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self) -> tuple[int, Node]:
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child  # type: ignore[return-value]

    def expand(self, state: np.ndarray, to_play: int, action_probs: np.ndarray) -> None:
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(
                    prior=prob, to_play=self.to_play * -1, parent=self
                )

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(
            self.state.__str__(), prior, self.visit_count, self.value()
        )


class MCTS:
    def __init__(self, game: BaseGame, model: BaseModel, num_simulations: int):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations

    def run(self, state: np.ndarray, to_play: int) -> Node:
        return self.run_batch([state], [to_play])[0]

    def run_batch(self, states: list[np.ndarray], to_plays: list[int]) -> list[Node]:
        roots: list[Node] = []
        # Initial batch prediction for roots
        action_probs_batch, _ = self.model.predict(states)

        for state, to_play, action_probs in zip(states, to_plays, action_probs_batch):
            root = Node(0, to_play)
            # EXPAND root
            valid_moves = self.game.get_valid_moves(state)
            action_probs = action_probs * valid_moves  # mask invalid moves
            action_probs /= np.sum(action_probs)
            root.expand(state, to_play, action_probs)
            roots.append(root)

        for _ in range(self.num_simulations):
            # Collect leaf nodes for batch prediction
            leaves: list[tuple[Node, np.ndarray]] = []

            for root in roots:
                node = root
                # SELECT
                while node.expanded():
                    action, node = node.select_child()

                parent: Node = node.parent  # type: ignore[assignment]
                state: np.ndarray = parent.state  # type: ignore[assignment]

                # Now we're at a leaf node and we would like to expand
                # Players always play from their own perspective
                next_state, _ = self.game.get_next_state(state, player=1, action=action)
                # Get the board from the perspective of the other player
                next_state = self.game.get_canonical_board(next_state, player=-1)

                # The value of the new state from the perspective of the other player
                value = self.game.get_reward_for_player(next_state, player=1)
                if value is None:
                    # EXPAND
                    # If game hasn't ended, add to batch prediction list
                    leaves.append((node, next_state))
                else:
                    # If game has ended, backpropagate immediately
                    self.backpropagate(node, value, parent.to_play * -1)

            if len(leaves) > 0:
                # Batch predict for all leaf nodes
                next_states = [leaf[1] for leaf in leaves]
                action_probs_batch, values_batch = self.model.predict(next_states)

                # Process predictions and expand nodes
                for leaf, action_probs, value in zip(
                    leaves, action_probs_batch, values_batch
                ):
                    node, next_state = leaf
                    parent = node.parent  # type: ignore[assignment]
                    valid_moves = self.game.get_valid_moves(next_state)
                    action_probs = action_probs * valid_moves  # mask invalid moves
                    action_probs /= np.sum(action_probs)
                    node.expand(next_state, parent.to_play * -1, action_probs)
                    self.backpropagate(node, value, parent.to_play * -1)

        return roots

    def run_single(self, state: np.ndarray, to_play: int) -> Node:
        root = Node(0, to_play)

        # EXPAND root
        [action_probs], _ = self.model.predict([state])
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        root.expand(state, to_play, action_probs)

        for _ in range(self.num_simulations):
            node = root

            # SELECT
            while node.expanded():
                action, node = node.select_child()

            parent: Node = node.parent  # type: ignore[assignment]
            state = parent.state  # type: ignore[assignment]

            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_reward_for_player(next_state, player=1)
            if value is None:
                # If the game has not ended:
                # EXPAND
                [action_probs], [value] = self.model.predict([next_state])
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)

            self.backpropagate(node, value, parent.to_play * -1)

        return root

    def backpropagate(self, leaf_node: Node, value: float, to_play: int):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        node = leaf_node
        while node is not None:
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            node = node.parent
