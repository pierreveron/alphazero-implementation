import torch
import torch.nn.functional as F
from simulator.game.connect import State  # type: ignore[import]
from torch import Tensor, nn

from alphazero_implementation.models.model import ActionPolicy, Model, Value


class NeuralNetwork(nn.Module, Model):
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

    def predict(self, states: list[State]) -> tuple[list[ActionPolicy], list[Value]]:
        # Convert states to tensor
        x: Tensor = torch.FloatTensor([state.to_input() for state in states])

        # Get predictions
        with torch.no_grad():
            policy_outputs, value_outputs = self.forward(x)

        action_policies: list[ActionPolicy] = []
        values: list[Value] = []
        for i, state in enumerate(states):
            for action in state.actions:
                policy_output = policy_outputs[i]
                value_output = value_outputs[i]
                action_policies.append({action: policy_output.tolist()})
                values.append(value_output.tolist())

        # Convert to lists of floats
        # policies = policy_outputs.tolist()
        # values = value_outputs.squeeze(-1).tolist()

        return action_policies, values
