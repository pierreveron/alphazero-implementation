from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from simulator.game.connect import State  # type: ignore[import]
from torch import Tensor, nn

from alphazero_implementation.models.model import ActionPolicy, Model, Value


class NeuralNetwork(nn.Module):
    """
    A neural network model that predicts the optimal action for a given game state.
    This model implements both nn.Module for PyTorch functionality and the custom Model interface.
    It uses convolutional layers to process the game state and fully connected layers to output action probabilities.
    """

    def __init__(self, input_shape: tuple[int, int, int, int], num_actions: int):
        super(NeuralNetwork, self).__init__()  # type: ignore[call-arg]
        self.conv1: nn.Conv2d = nn.Conv2d(input_shape[1], 32, kernel_size=3, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3: nn.Conv2d = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Calculate the size of the flattened features
        self.flat_features: int = 64 * input_shape[2] * input_shape[3]

        self.fc1: nn.Linear = nn.Linear(self.flat_features, 256)
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flat_features)
        x = F.relu(self.fc1(x))

        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))

        return policy, value


class AlphaZeroLightningModule(pl.LightningModule, Model):
    def __init__(self, input_shape: tuple[int, int, int, int], num_actions: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuralNetwork(input_shape, num_actions)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        states, policy_targets, value_targets = batch
        policy_outputs, value_outputs = self(states)

        policy_loss = F.cross_entropy(policy_outputs, policy_targets)
        value_loss = F.mse_loss(value_outputs.squeeze(), value_targets)
        loss = policy_loss + value_loss

        self.log("train_policy_loss", policy_loss)  # type: ignore[call-arg]
        self.log("train_value_loss", value_loss)  # type: ignore[call-arg]
        self.log("train_loss", loss)  # type: ignore[call-arg]

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)  # type: ignore[no-untyped-call]

    def predict(self, states: list[State]) -> tuple[list[ActionPolicy], list[Value]]:
        # Convert states to tensor
        x: Tensor = torch.FloatTensor([state.to_input() for state in states])

        # Get predictions
        with torch.no_grad():
            policy_outputs, value_outputs = self(x)

        action_policies: list[ActionPolicy] = []
        values: list[Value] = []
        for i, state in enumerate(states):
            policy_output = policy_outputs[i]
            value_output = value_outputs[i]
            action_policy = {
                action: float(policy_output[j])
                for j, action in enumerate(state.actions)
            }
            action_policies.append(action_policy)
            values.append(float(value_output.item()))

        return action_policies, values
