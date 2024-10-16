import numpy as np
from simulator.game.connect import (
    Action,  # type: ignore[import]
    State,  # type: ignore[attr-defined]
)

from alphazero_implementation.alphazero.trainer import GameHistory
from alphazero_implementation.mcts.mcgs import Node, select_action_according_to_puct
from alphazero_implementation.models.model import ActionPolicy, Model


class MCTSAgent:
    def __init__(
        self, model: Model, self_play_count: int, num_simulations_per_self_play: int
    ):
        self.model = model
        self.self_play_count = self_play_count
        self.num_simulations_per_self_play = num_simulations_per_self_play

    def sample_action_from_policy(self, policy: ActionPolicy) -> Action:
        index = np.random.choice(len(policy), p=list(policy.values()))
        return list(policy.keys())[index]

    def run_self_plays(self, initial_state: State) -> list[GameHistory]:
        game_histories: list[GameHistory] = [[] for _ in range(self.self_play_count)]
        games: list[State | None] = [initial_state for _ in range(self.self_play_count)]

        while any(
            game is not None for game in games
        ):  # while not all(game is None for game in games):
            print(f"Games: {games}")
            roots: list[Node | None] = [
                Node(game_state=state) for state in games if state is not None
            ]

            # Monte Carlo Tree Search / Graph Search
            for _ in range(self.num_simulations_per_self_play):
                print(f"Roots: {roots}")
                leaf_nodes: list[tuple[Node, list[tuple[Node, Action]]]] = []
                nodes_by_state_list: list[dict[State, Node]] = [
                    {root.game_state: root} for root in roots if root is not None
                ]

                for root in roots:
                    print(f"Root: {root}")
                    if root is None:
                        continue
                    node: Node = root
                    path: list[tuple[Node, Action]] = []

                    while True:
                        print(f"Node: {node}")
                        if node.game_state.has_ended:
                            node.U = node.game_state.reward  # type: ignore[attr-defined]
                            break
                        elif node.N == 0:  # New node not yet visited
                            leaf_nodes.append((node, path))
                            print(f"Leaf nodes: {leaf_nodes}")
                            break
                        else:
                            # 1. Selection
                            action: Action = select_action_according_to_puct(node)
                            print(f"Action: {action}")
                            if action not in node.children_and_edge_visits:
                                # 2. Expansion
                                new_game_state: State = action.sample_next_state()
                                if (
                                    new_game_state
                                    in nodes_by_state_list[roots.index(root)]
                                ):
                                    child: Node = nodes_by_state_list[
                                        roots.index(root)
                                    ][new_game_state]
                                    node.children_and_edge_visits[action] = (child, 0)
                                else:
                                    new_node: Node = Node(game_state=new_game_state)
                                    node.children_and_edge_visits[action] = (
                                        new_node,
                                        0,
                                    )
                                    nodes_by_state_list[roots.index(root)][
                                        new_game_state
                                    ] = new_node
                                child: Node = node.children_and_edge_visits[action][0]
                                path.append((node, action))
                                node = child
                                break
                            else:
                                child, _ = node.children_and_edge_visits[action]
                                path.append((node, action))
                                node = child

                # Evaluate leaf nodes in batch
                if leaf_nodes:
                    print(f"Leaf nodes: {leaf_nodes}")
                    states_to_evaluate: list[State] = [
                        node.game_state for node, _ in leaf_nodes
                    ]

                    policies, values = self.model.predict(states_to_evaluate)
                    print(f"Policies: {policies}")
                    print(f"Values: {values}")

                    for i, (node, path) in enumerate(leaf_nodes):
                        node.action_policy = policies[i]
                        node.U = values[i]

                    # Backpropagation
                    print("Backpropagation")
                    for node, path in leaf_nodes:
                        for parent, action in reversed(path):
                            child, edge_visits = parent.children_and_edge_visits[action]
                            parent.children_and_edge_visits[action] = (
                                child,
                                edge_visits + 1,
                            )

                            children_and_edge_visits = (
                                parent.children_and_edge_visits.values()
                            )
                            parent.N = 1 + sum(
                                edge_visits
                                for (_, edge_visits) in children_and_edge_visits
                            )
                            parent.Q = (1 / parent.N) * (
                                parent.U[parent.game_state.player]
                                + sum(
                                    child.Q * edge_visits
                                    for (child, edge_visits) in children_and_edge_visits
                                )
                            )

            # Calculate improved policies and choose actions
            for root_index, root in enumerate(roots):
                print(f"Root index: {root_index}")
                print(f"Root: {root}")
                if root is None:
                    continue
                state = root.game_state

                visits_per_action: dict[Action, float] = {}
                sum_visits = 0
                for action in state.actions:
                    if action in root.children_and_edge_visits:
                        visits_per_action[action] = root.children_and_edge_visits[
                            action
                        ][1]
                        sum_visits += root.children_and_edge_visits[action][1]

                if sum_visits > 0:
                    improved_policy: ActionPolicy = {
                        action: action_visits / sum_visits
                        for action, action_visits in visits_per_action.items()
                    }
                else:
                    # If sum_visits is 0, use a uniform distribution on valid actions
                    improved_policy: ActionPolicy = {
                        action: 1 / len(state.actions) for action in state.actions
                    }

                game_histories[root_index].append(
                    (
                        state,
                        improved_policy,
                        [0 for _ in range(state.config.num_players)],
                    )
                )

                # Choose action based on the improved policy
                chosen_action: Action = self.sample_action_from_policy(improved_policy)
                new_state: State = chosen_action.sample_next_state()

                if not new_state.has_ended:
                    games[root_index] = new_state
                else:
                    games[root_index] = None
                    final_reward = new_state.reward  # type: ignore[attr-defined]
                    # Backpropagate the final reward to all states in this game's history
                    # Start from the last state and go backwards
                    for j in range(len(game_histories[root_index]) - 1, -1, -1):
                        game_histories[root_index][j] = game_histories[root_index][j][
                            :2
                        ] + (final_reward.tolist(),)

        return game_histories
