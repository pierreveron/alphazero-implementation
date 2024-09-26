from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from numpy.typing import NDArray
from simulator.game.connect import State  # type: ignore[import]
from torch import nn, optim

from alphazero_implementation.mcts.mcgs import Node, perform_one_playout
from alphazero_implementation.models.neural_network import NeuralNetwork


class AlphaZeroMCGS:
    def __init__(self, neural_network: NeuralNetwork, num_simulations: int = 800):
        self.neural_network = neural_network
        self.num_simulations = num_simulations

    def search(self, state: State) -> list[float]:
        # Create a local dictionary for nodes_by_hash
        nodes_by_hash: dict[int, Node] = {}

        # Create the root node
        root = Node(game_state=state)
        nodes_by_hash[root.hash] = root

        # Perform MCGS simulations
        for _ in range(self.num_simulations):
            perform_one_playout(root, nodes_by_hash)

        # Calculate the improved policy
        visits = np.array(
            [
                root.children_and_edge_visits[action][1]
                if action in root.children_and_edge_visits
                else 0
                for action in state.actions
            ]
        )
        improved_policy: NDArray[np.float64] = visits / np.sum(visits)

        return improved_policy.tolist()

    def self_play(self, initial_state: State) -> list[tuple[State, list[float], float]]:
        trajectory: list[tuple[State, list[float], float]] = []
        state = initial_state

        while not state.has_ended:
            improved_policy = self.search(state)
            trajectory.append(
                (state, improved_policy, 0)
            )  # 0 is a placeholder for the value

            # Choose action based on the improved policy
            action = np.random.choice(len(state.actions), p=improved_policy)
            state = state.actions[action].sample_next_state()

        # Update the values in the trajectory
        final_reward = state.reward  # type: ignore[attr-defined]
        for i in range(len(trajectory) - 1, -1, -1):
            state, policy, _ = trajectory[i]
            trajectory[i] = (state, policy, final_reward[state.player])

        return trajectory

    def parallel_self_play(
        self,
        batch_size: int,
        initial_state: State,
    ) -> list[list[tuple[State, list[float], float]]]:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.self_play, initial_state)
                for _ in range(batch_size)
            ]
            results = [future.result() for future in futures]
        return results

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

        for iteration in range(num_iterations):
            # Generate self-play games using ThreadPoolExecutor
            trajectories = self.parallel_self_play(self_play_batch_size, initial_state)
            training_data: list[tuple[State, list[float], float]] = [
                item for sublist in trajectories for item in sublist
            ]

            # Train the neural network
            states, policies, values = zip(*training_data)

            # Convert data to tensors
            state_inputs = torch.FloatTensor([state.to_input() for state in states])
            policy_targets = torch.FloatTensor(policies)
            value_targets = torch.FloatTensor(values)

            # Training loop
            num_epochs = 10
            for epoch in range(num_epochs):
                # Forward pass
                policy_outputs, value_outputs = self.neural_network(state_inputs)

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
