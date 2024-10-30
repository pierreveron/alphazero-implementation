import numpy as np
from simulator.game.connect import Action, State  # type: ignore[import]

from alphazero_implementation.helpers.timeit import timeit
from alphazero_implementation.mcts.mcgs import Node, select_action_according_to_puct
from alphazero_implementation.models.model import ActionPolicy, Model

# GameHistory represents the trajectory of a single game
# It is a list of tuples, where each tuple contains:
# - State: The game state at that point
# - list[float]: The improved policy (action probabilities) for that state
# - list[float]: The value (expected outcome) for each player at that state
GameHistoryItem = tuple[State, ActionPolicy, list[float]]
GameHistory = list[GameHistoryItem]


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
    def run(self) -> GameHistory:
        if self.parallel_mode:
            return self.run_in_parallel()
        else:
            return self.run_sequentially()

    def run_sequentially(self) -> GameHistory:
        batch_data: GameHistory = []

        for _ in range(self.num_episodes):
            game_data = self.self_play(self.initial_state)
            batch_data.extend(game_data)

        return batch_data

    def run_in_parallel(self) -> GameHistory:
        batch_data: GameHistory = []

        data = self.batch_self_play(self.initial_state)

        for game_data in data:
            batch_data.extend(game_data)
        return batch_data

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

    def self_play(self, initial_state: State) -> GameHistory:
        game_history: GameHistory = []
        current_node = Node(game_state=initial_state)
        nodes_by_state: dict[State, Node] = {initial_state: current_node}
        while not current_node.game_state.has_ended:
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
            game_history.append(
                (
                    current_node.game_state,
                    policy,
                    [0.0] * initial_state.config.num_players,
                )
            )

            # Choose action based on policy and progress to next state
            action = self.sample_action_from_policy(policy)
            current_node = current_node.children_and_edge_visits[action][0]

        # Determine game outcome (e.g., +1 for win, -1 for loss, 0 for draw)
        outcome = current_node.game_state.reward  # type: ignore[attr-defined]
        for i in range(len(game_history)):
            state, policy, _ = game_history[i]
            game_history[i] = (state, policy, outcome.tolist())  # type: ignore[attr-defined]

        return game_history

    def mcts_search_recursive(self, node: Node, nodes_by_state: dict[State, Node]):
        if node.game_state.has_ended:
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
            if node.game_state.has_ended:
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

    def batch_self_play(self, initial_state: State) -> list[GameHistory]:
        game_histories: list[GameHistory] = [[]] * self.num_episodes
        games: list[State | None] = [initial_state] * self.num_episodes

        num_players = initial_state.config.num_players

        while any(
            game is not None for game in games
        ):  # while not all(game is None for game in games):
            roots: list[Node | None] = [
                Node(game_state=state) if state is not None else None for state in games
            ]
            assert len(game_histories) == len(roots) == len(games)

            # Monte Carlo Tree Search / Graph Search
            for _ in range(self.simulations_per_episode):
                leaf_nodes: list[tuple[Node, list[tuple[Node, Action]]]] = []
                nodes_by_state_list: list[dict[State, Node]] = [
                    {root.game_state: root} for root in roots if root is not None
                ]

                for root_index, root in enumerate(roots):
                    if root is None:
                        continue
                    node: Node = root
                    path: list[tuple[Node, Action]] = []

                    while True:
                        if node.game_state.has_ended:
                            node.utility_values = node.game_state.reward.tolist()  # type: ignore[attr-defined]
                            break
                        elif node.visit_count == 0:  # New node not yet visited
                            leaf_nodes.append((node, path))
                            break
                        else:
                            action: Action = select_action_according_to_puct(node)
                            if action not in node.children_and_edge_visits:
                                new_game_state: State = action.sample_next_state()
                                if new_game_state in nodes_by_state_list[root_index]:
                                    child = nodes_by_state_list[root_index][
                                        new_game_state
                                    ]
                                else:
                                    child = Node(game_state=new_game_state)
                                    nodes_by_state_list[root_index][new_game_state] = (
                                        child
                                    )

                                node.children_and_edge_visits[action] = (child, 0)
                            else:
                                child = node.children_and_edge_visits[action][0]
                            path.append((node, action))
                            node = child

                    # Now backpropagate the values along the path
                    # First, set N and Q for the leaf node
                    node.visit_count = 1
                    node.cumulative_value = node.utility_values[node.game_state.player]

                # Evaluate leaf nodes in batch
                if leaf_nodes:
                    states_to_evaluate: list[State] = [
                        node.game_state for node, _ in leaf_nodes
                    ]

                    policies, values = self.model.predict(states_to_evaluate)

                    for i, (node, path) in enumerate(leaf_nodes):
                        node.action_policy = policies[i]
                        node.utility_values = values[i]

                    # Backpropagation
                    for _, path in leaf_nodes:
                        self.backpropagate(path)

            # Calculate improved policies and choose actions
            for root_index, root in enumerate(roots):
                if root is None:
                    continue

                improved_policy = self.calculate_improved_policy(root)

                state = root.game_state
                game_histories[root_index].append(
                    (
                        state,
                        improved_policy,
                        [0.0] * num_players,
                    )
                )

                # Choose action based on the improved policy
                chosen_action: Action = self.sample_action_from_policy(improved_policy)
                new_state: State = chosen_action.sample_next_state()

                if not new_state.has_ended:
                    games[root_index] = new_state
                else:
                    games[root_index] = None
                    final_reward = new_state.reward.tolist()  # type: ignore[attr-defined]

                    # Backpropagate the final reward to all states in this game's history
                    game_history = game_histories[root_index]
                    for i in range(len(game_history)):
                        state, improved_policy, _ = game_history[i]
                        game_history[i] = (state, improved_policy, final_reward)

        return game_histories
