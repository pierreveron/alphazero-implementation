import lightning as L
import numpy as np
import torch
from lightning.pytorch.tuner import Tuner  # type: ignore[import]
from numpy.typing import NDArray
from simulator.game.connect import Action, State  # type: ignore[import]
from torch.utils.data import DataLoader, TensorDataset

from alphazero_implementation.mcts.mcgs import (
    Node,
    select_action_according_to_puct,
)

# GameHistory represents the trajectory of a single game
# It is a list of tuples, where each tuple contains:
# - State: The game state at that point
# - list[float]: The improved policy (action probabilities) for that state
# - list[float]: The value (expected outcome) for each player at that state
GameHistory = list[tuple[State, list[float], list[float]]]


class AlphaZeroTrainer:
    def __init__(self, model: L.LightningModule, num_simulations: int = 800):
        self.model = model
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

                    policies, values = self.model.predict(states_to_evaluate)

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
        training_data: GameHistory = []

        trainer = L.Trainer(
            max_epochs=10,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )
        tuner = Tuner(trainer)

        for iteration in range(num_iterations):
            trajectories = self.parallel_self_play(self_play_batch_size, initial_state)
            training_data.extend([item for sublist in trajectories for item in sublist])
            # TODO: remove old training data

            states, policies, values = zip(*training_data)
            state_inputs = torch.FloatTensor([state.grid for state in states])
            policy_targets = torch.FloatTensor(policies)
            value_targets = torch.FloatTensor(values)

            dataset = TensorDataset(state_inputs, policy_targets, value_targets)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            tuner.lr_find(self.model)

            trainer.fit(self.model, dataloader)

            print(f"Iteration [{iteration+1}/{num_iterations}] completed!")

        print("Training completed!")
