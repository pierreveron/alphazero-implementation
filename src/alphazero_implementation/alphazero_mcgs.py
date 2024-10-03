import numpy as np
import torch
from numpy.typing import NDArray
from simulator.game.connect import Action, State  # type: ignore[import]
from torch import nn, optim

from alphazero_implementation.mcts.mcgs import (
    Node,
    select_action_according_to_puct,
)
from alphazero_implementation.models.neural_network import NeuralNetwork

# GameHistory represents the trajectory of a single game
# It is a list of tuples, where each tuple contains:
# - State: The game state at that point
# - list[float]: The improved policy (action probabilities) for that state
# - list[float]: The value (expected outcome) for each player at that state
GameHistory = list[tuple[State, list[float], list[float]]]


class AlphaZeroMCGS:
    def __init__(self, neural_network: NeuralNetwork, num_simulations: int = 800):
        self.neural_network = neural_network
        self.num_simulations = num_simulations

    def parallel_self_play(
        self,
        batch_size: int,
        initial_state: State,
    ) -> list[GameHistory]:
        game_histories: list[GameHistory] = [[] for _ in range(batch_size)]
        games: list[State | None] = [initial_state for _ in range(batch_size)]

        while any(
            game is not None for game in games
        ):  # while not all(game is None for game in games):
            roots: list[Node | None] = [
                Node(game_state=state) for state in games if state is not None
            ]
            nodes_by_state_list: list[dict[State, Node]] = [
                {root.game_state: root} for root in roots if root is not None
            ]

            # Monte Carlo Tree Search / Graph Search
            for _ in range(self.num_simulations):
                leaf_nodes: list[tuple[Node, list[tuple[Node, Action]]]] = []

                for root in roots:
                    if root is None:
                        continue
                    node: Node = root
                    path: list[tuple[Node, Action]] = []

                    while True:
                        if node.game_state.has_ended:
                            node.U = node.game_state.reward  # type: ignore[attr-defined]
                            break
                        elif node.N == 0:  # New node not yet visited
                            leaf_nodes.append((node, path))
                            break
                        else:
                            # 1. Selection
                            action: Action = select_action_according_to_puct(node)
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
                    states_to_evaluate: list[State] = [
                        node.game_state for node, _ in leaf_nodes
                    ]

                    policies, values = self.neural_network.predict(states_to_evaluate)

                    for i, (node, path) in enumerate(leaf_nodes):
                        node.action_policy = policies[i]
                        node.U = values[i]

                    # Backpropagation
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
                if root is None:
                    continue
                state = root.game_state
                visits = np.array(
                    [
                        root.children_and_edge_visits[action][1]
                        if action in root.children_and_edge_visits
                        else 0
                        for action in state.actions
                    ],
                    dtype=np.float64,
                )
                improved_policy: NDArray[np.float64] = visits / np.sum(visits)
                game_histories[root_index].append(
                    (
                        state,
                        improved_policy.tolist(),
                        [0 for _ in range(state.config.num_players)],
                        # 0 is a placeholder for the value
                    )
                )

                # Choose action based on the improved policy
                action_index = np.random.choice(len(state.actions), p=improved_policy)
                chosen_action: Action = state.actions[action_index]
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

    def train(
        self,
        num_iterations: int,
        self_play_batch_size: int,
        initial_state: State,
    ):
        # Replace BCELoss with CrossEntropyLoss for policy and MSELoss for value
        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.neural_network.parameters(), lr=0.01)  # type: ignore[no-untyped-call]
        training_data: list[tuple[State, list[float], list[float]]] = []

        for iteration in range(num_iterations):
            # Generate self-play games using ThreadPoolExecutor
            trajectories = self.parallel_self_play(self_play_batch_size, initial_state)
            training_data.extend([item for sublist in trajectories for item in sublist])
            # TODO: remove old training data

            # Train the neural network
            states, policies, values = zip(*training_data)

            # Convert data to tensors
            state_inputs = torch.FloatTensor([state.to_input() for state in states])
            policy_targets = torch.FloatTensor(policies)
            value_targets = torch.FloatTensor(values)

            # Training loop
            num_epochs = 10
            for epoch in range(num_epochs):
                # Shuffle the data
                indices = torch.randperm(len(state_inputs))
                state_inputs = state_inputs[indices]
                policy_targets = policy_targets[indices]
                value_targets = value_targets[indices]
                batch_size = 10
                for i in range(0, len(state_inputs), batch_size):
                    # Forward pass
                    policy_outputs, value_outputs = self.neural_network(
                        state_inputs[i : i + batch_size]
                    )

                    # Calculate losses
                    policy_loss = policy_criterion(policy_outputs, policy_targets)
                    value_loss = value_criterion(value_outputs.squeeze(), value_targets)
                    loss = policy_loss + value_loss

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()  # type: ignore[no-untyped-call]

                    print(
                        f"Iteration [{iteration+1}/{num_iterations}], Epoch [{epoch+1}/{num_epochs}], "
                        f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, "
                        f"Total Loss: {loss.item():.4f}"
                    )

        print("Training completed!")

    # def get_best_action(self, state: State) -> Action:
    #     improved_policy = self.search(state)
    #     return state.actions[np.argmax(improved_policy)]
