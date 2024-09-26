import torch.nn.functional as F
from torch import Tensor, nn


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
        self.fc2: nn.Linear = nn.Linear(256, num_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flat_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    # def predict(self, states: list[State]) -> list[tuple[ActionPolicy, Value]]:
    #     # Convert states to tensor
    #     x: Tensor = torch.FloatTensor([state.to_input() for state in states])

    #     # Get predictions
    #     with torch.no_grad():
    #         outputs: Tensor = self.forward(x)

    #     # Convert outputs to actions
    #     action_indices = outputs.argmax(dim=1).tolist()
    #     actions = [Action(int(index)) for index in action_indices]

    #     # For now, we're still returning 0 for the value predictions
    #     values = [0.0] * len(states)

    #     return actions, values
