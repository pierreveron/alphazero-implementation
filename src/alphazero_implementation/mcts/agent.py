from typing import Generator

import numpy as np
from simulator.game.connect import Action, State  # type: ignore[import]

from alphazero_implementation.alphazero.types import Episode, Sample
from alphazero_implementation.helpers.timeit import timeit
from alphazero_implementation.mcts.mcgs import Node, select_action_according_to_puct
from alphazero_implementation.models.model import ActionPolicy, Model


class MCTSAgent:
    def __init__(
        self,
        model: Model,
        num_episodes: int,
        simulations_per_episode: int,
        initial_state: State,
        parallel_mode: bool = False,
    ):
        self.model = model
        self.num_episodes = num_episodes
        self.simulations_per_episode = simulations_per_episode
        self.initial_state = initial_state
        self.parallel_mode = parallel_mode

    @timeit
    def run(self) -> list[Episode]:
        if self.parallel_mode:
            return self.run_in_parallel()
        else:
            return self.run_sequentially()

    def run_sequentially(self) -> list[Episode]:
        episodes: list[Episode] = []

        for _ in range(self.num_episodes):
            episodes.append(self.self_play(self.initial_state))

        return episodes

    def run_in_parallel(self) -> list[Episode]:
        episodes = self.batch_self_play(self.initial_state)
        return episodes

    def sample_action_from_policy(self, policy: ActionPolicy) -> Action:
        index = np.random.choice(len(policy), p=list(policy.values()))
        return list(policy.keys())[index]

    def calculate_improved_policy(self, root: Node) -> ActionPolicy:
        visits_per_action = {
            action: root.children_and_edge_visits[action][1]
            for action in root.game_state.actions
            if action in root.children_and_edge_visits
        }
        sum_visits = sum(visits_per_action.values())

        if sum_visits > 0:
            return {
                action: visits / sum_visits
                for action, visits in visits_per_action.items()
            }
        else:
            return {
                action: 1 / len(root.game_state.actions)
                for action in root.game_state.actions
            }

    def select_next_node(self, node: Node) -> Node:
        """Select the most visited child node and return its action and node."""
        action = max(
            node.children_and_edge_visits.items(),
            key=lambda x: x[1][1],  # x[1][1] is the edge visit count
        )[0]
        next_node = node.children_and_edge_visits[action][0]
        return next_node

    def self_play(self, initial_state: State) -> Episode:
        episode = Episode()
        current_node = Node(game_state=initial_state)
        nodes_by_state: dict[State, Node] = {initial_state: current_node}
        while not current_node.is_terminal:
            # Run MCTS to get policy
            for _ in range(self.simulations_per_episode):
                # self.mcts_search_recursive(current_node, nodes_by_state)
                self.mcts_search(current_node, nodes_by_state)

            # Get policy as a probability distribution
            policy = {
                action: edge_visits / (current_node.visit_count - 1)
                for action, (
                    _,
                    edge_visits,
                ) in current_node.children_and_edge_visits.items()
            }

            # Collect data: (state, policy, outcome) where outcome will be assigned later
            episode.add_sample(
                Sample(
                    current_node.game_state,
                    policy,
                    [0.0] * initial_state.config.num_players,
                )
            )

            current_node = self.select_next_node(current_node)

        # Determine game outcome (e.g., +1 for win, -1 for loss, 0 for draw)
        outcome = current_node.game_state.reward  # type: ignore[attr-defined]
        episode_history = episode.samples
        for i in range(len(episode_history)):
            episode_history[i].value = outcome.tolist()  # type: ignore[attr-defined]

        return episode

    def mcts_search_recursive(self, node: Node, nodes_by_state: dict[State, Node]):
        if node.is_terminal:
            node.utility_values = node.game_state.reward.tolist()  # type: ignore[attr-defined]
        elif node.visit_count == 0:  # New node not yet visited
            policies, values = self.model.predict([node.game_state])
            node.action_policy = policies[0]
            node.utility_values = values[0]
        else:
            action = select_action_according_to_puct(node)
            if action not in node.children_and_edge_visits:
                new_game_state = action.sample_next_state()

                if new_game_state in nodes_by_state:
                    child = nodes_by_state[new_game_state]
                else:
                    child = Node(game_state=new_game_state)
                    nodes_by_state[new_game_state] = child
                node.children_and_edge_visits[action] = (child, 0)

            child, edge_visits = node.children_and_edge_visits[action]
            self.mcts_search_recursive(child, nodes_by_state)
            node.children_and_edge_visits[action] = (child, edge_visits + 1)

        # Update statistics
        children_and_edge_visits = node.children_and_edge_visits.values()
        node.visit_count = 1 + sum(
            edge_visits for (_, edge_visits) in children_and_edge_visits
        )
        node.cumulative_value = (1 / node.visit_count) * (
            node.utility_values[node.game_state.player]
            + sum(
                child.cumulative_value * edge_visits
                for (child, edge_visits) in children_and_edge_visits
            )
        )

    def backpropagate(self, path: list[tuple[Node, Action]]):
        for parent, action in reversed(path):
            # Update edge visits
            child, edge_visits = parent.children_and_edge_visits[action]
            parent.children_and_edge_visits[action] = (child, edge_visits + 1)

            # Update visit count and cumulative value
            children_and_edge_visits = parent.children_and_edge_visits.values()
            parent.visit_count = 1 + sum(
                edge_visits for (_, edge_visits) in children_and_edge_visits
            )
            parent.cumulative_value = (1 / parent.visit_count) * (
                parent.utility_values[parent.game_state.player]
                + sum(
                    child.cumulative_value * edge_visits
                    for (child, edge_visits) in children_and_edge_visits
                )
            )

    def mcts_search(self, root: Node, nodes_by_state: dict[State, Node]):
        node = root
        path: list[tuple[Node, Action]] = []

        while True:
            if node.is_terminal:
                node.utility_values = node.game_state.reward.tolist()  # type: ignore[attr-defined]
                break
            elif node.visit_count == 0:  # New node not yet visited
                policies, values = self.model.predict([node.game_state])
                node.action_policy = policies[0]
                node.utility_values = values[0]
                break
            else:
                action = select_action_according_to_puct(node)
                if action not in node.children_and_edge_visits:
                    new_game_state = action.sample_next_state()

                    if new_game_state in nodes_by_state:
                        child = nodes_by_state[new_game_state]
                    else:
                        child = Node(game_state=new_game_state)
                        nodes_by_state[new_game_state] = child

                    node.children_and_edge_visits[action] = (child, 0)
                else:
                    child = node.children_and_edge_visits[action][0]
                path.append((node, action))
                node = child

        # Now backpropagate the values along the path
        # First, set N and Q for the leaf node
        node.visit_count = 1
        node.cumulative_value = node.utility_values[node.game_state.player]

        # Backpropagate from the leaf node up to the root
        self.backpropagate(path)

    def batch_self_play(self, initial_state: State) -> list[Episode]:
        num_players = initial_state.config.num_players
        episodes = [Episode() for _ in range(self.num_episodes)]
        current_nodes: list[Node] = [
            Node(game_state=initial_state) for _ in range(self.num_episodes)
        ]

        nodes_by_state_list: list[dict[State, Node]] = [
            {node.game_state: node} for node in current_nodes
        ]

        while any(not node.is_terminal for node in current_nodes):
            # Monte Carlo Tree Search / Graph Search
            for _ in range(self.simulations_per_episode):
                leaf_nodes: list[tuple[Node, list[tuple[Node, Action]]]] = []

                for current_node_index, current_node in enumerate(current_nodes):
                    if current_node.is_terminal:
                        continue

                    node: Node = current_node
                    path: list[tuple[Node, Action]] = []
                    nodes_by_state = nodes_by_state_list[current_node_index]

                    while True:
                        if node.is_terminal:
                            node.utility_values = node.game_state.reward.tolist()  # type: ignore[attr-defined]
                            # Add terminal nodes to leaf_nodes for backpropagation
                            leaf_nodes.append((node, path))
                            break
                        elif node.visit_count == 0:  # New node not yet visited
                            leaf_nodes.append((node, path))
                            break
                        else:
                            action = select_action_according_to_puct(node)
                            if action not in node.children_and_edge_visits:
                                new_game_state = action.sample_next_state()

                                if new_game_state in nodes_by_state:
                                    child = nodes_by_state[new_game_state]
                                else:
                                    child = Node(game_state=new_game_state)
                                    nodes_by_state[new_game_state] = child

                                node.children_and_edge_visits[action] = (child, 0)
                            else:
                                child = node.children_and_edge_visits[action][0]
                            path.append((node, action))
                            node = child

                # Evaluate leaf nodes in batch
                if leaf_nodes:
                    non_terminal_nodes = [
                        (node, path)
                        for node, path in leaf_nodes
                        if not node.is_terminal
                    ]

                    # Handle non-terminal nodes with neural network
                    if non_terminal_nodes:
                        states_to_evaluate = [
                            node.game_state for node, _ in non_terminal_nodes
                        ]
                        policies, values = self.model.predict(states_to_evaluate)

                        for i, (node, _) in enumerate(non_terminal_nodes):
                            node.action_policy = policies[i]
                            node.utility_values = values[i]

                    # Set visit count and cumulative value for all leaf nodes
                    for node, _ in leaf_nodes:
                        node.visit_count = 1
                        node.cumulative_value = node.utility_values[
                            node.game_state.player
                        ]

                    # Backpropagate all paths
                    for _, path in leaf_nodes:
                        self.backpropagate(path)

            # Calculate improved policies and choose actions
            for current_node_index, current_node in enumerate(current_nodes):
                if current_node.is_terminal:
                    continue

                # policy = self.calculate_improved_policy(current_node)
                # Get policy as a probability distribution
                policy = {
                    action: edge_visits / (current_node.visit_count - 1)
                    for action, (
                        _,
                        edge_visits,
                    ) in current_node.children_and_edge_visits.items()
                }

                episodes[current_node_index].add_sample(
                    Sample(
                        current_node.game_state,
                        policy,
                        [0.0] * num_players,
                    )
                )

                new_current_node = self.select_next_node(current_node)
                current_nodes[current_node_index] = new_current_node

        for current_node_index, current_node in enumerate(current_nodes):
            # Determine game outcome (e.g., +1 for win, -1 for loss, 0 for draw)
            outcome = current_node.game_state.reward  # type: ignore[attr-defined]
            episode_history = episodes[current_node_index].samples
            for i in range(len(episode_history)):
                episode_history[i].value = outcome.tolist()  # type: ignore[attr-defined]

        return episodes

    def generate_episodes(self) -> Generator[Episode, None, None]:
        num_players = self.initial_state.config.num_players
        episodes = [Episode() for _ in range(self.num_episodes)]
        current_nodes: list[Node] = [
            Node(game_state=self.initial_state) for _ in range(self.num_episodes)
        ]

        nodes_by_state_list: list[dict[State, Node]] = [
            {node.game_state: node} for node in current_nodes
        ]

        while True:
            # Monte Carlo Tree Search / Graph Search
            for _ in range(self.simulations_per_episode):
                leaf_nodes: list[tuple[Node, list[tuple[Node, Action]]]] = []

                for current_node_index, current_node in enumerate(current_nodes):
                    if current_node.is_terminal:
                        continue

                    node: Node = current_node
                    path: list[tuple[Node, Action]] = []
                    nodes_by_state = nodes_by_state_list[current_node_index]

                    while True:
                        if node.is_terminal:
                            node.utility_values = node.game_state.reward.tolist()  # type: ignore[attr-defined]
                            # Add terminal nodes to leaf_nodes for backpropagation
                            leaf_nodes.append((node, path))
                            break
                        elif node.visit_count == 0:  # New node not yet visited
                            leaf_nodes.append((node, path))
                            break
                        else:
                            action = select_action_according_to_puct(node)
                            if action not in node.children_and_edge_visits:
                                new_game_state = action.sample_next_state()

                                if new_game_state in nodes_by_state:
                                    child = nodes_by_state[new_game_state]
                                else:
                                    child = Node(game_state=new_game_state)
                                    nodes_by_state[new_game_state] = child

                                node.children_and_edge_visits[action] = (child, 0)
                            else:
                                child = node.children_and_edge_visits[action][0]
                            path.append((node, action))
                            node = child

                # Evaluate leaf nodes in batch
                if leaf_nodes:
                    non_terminal_nodes = [
                        (node, path)
                        for node, path in leaf_nodes
                        if not node.is_terminal
                    ]

                    # Handle non-terminal nodes with neural network
                    if non_terminal_nodes:
                        states_to_evaluate = [
                            node.game_state for node, _ in non_terminal_nodes
                        ]
                        policies, values = self.model.predict(states_to_evaluate)

                        for i, (node, _) in enumerate(non_terminal_nodes):
                            node.action_policy = policies[i]
                            node.utility_values = values[i]

                    # Set visit count and cumulative value for all leaf nodes
                    for node, _ in leaf_nodes:
                        node.visit_count = 1
                        node.cumulative_value = node.utility_values[
                            node.game_state.player
                        ]

                    # Backpropagate all paths
                    for _, path in leaf_nodes:
                        self.backpropagate(path)

            # Calculate improved policies and choose actions
            for current_node_index, current_node in enumerate(current_nodes):
                if current_node.is_terminal:
                    continue

                # Get policy as a probability distribution
                policy = {
                    action: edge_visits / (current_node.visit_count - 1)
                    for action, (
                        _,
                        edge_visits,
                    ) in current_node.children_and_edge_visits.items()
                }

                episodes[current_node_index].add_sample(
                    Sample(
                        current_node.game_state,
                        policy,
                        [0.0] * num_players,
                    )
                )

                current_nodes[current_node_index] = self.select_next_node(current_node)

            for current_node_index, current_node in enumerate(current_nodes):
                if not current_node.is_terminal:
                    continue

                # Determine game outcome (e.g., +1 for win, -1 for loss, 0 for draw)
                outcome = current_node.game_state.reward  # type: ignore[attr-defined]
                episode_history = episodes[current_node_index].samples
                for i in range(len(episode_history)):
                    episode_history[i].value = outcome.tolist()  # type: ignore[attr-defined]

                yield episodes[current_node_index]
                # Replace with a new episode
                episodes[current_node_index] = Episode()
                current_nodes[current_node_index] = Node(game_state=self.initial_state)
                nodes_by_state_list[current_node_index] = {
                    self.initial_state: current_nodes[current_node_index]
                }
