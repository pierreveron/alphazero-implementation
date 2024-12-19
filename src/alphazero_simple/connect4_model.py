import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base_model import BaseModel


class Connect4Model(BaseModel):
    def __init__(
        self, board_size: tuple[int, int], action_size: int, device: torch.device
    ):
        super(Connect4Model, self).__init__()  # type: ignore[no-untyped-call]

        self.device = device
        self.rows, self.cols = board_size
        self.action_size = action_size

        # Input channels = 1 (just the board state)
        # Output channels = 32 (number of filters)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Calculate the size of the flattened features
        # After 3 conv layers with padding, spatial dimensions remain the same
        self.flat_size = 64 * self.rows * self.cols

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 128)

        # Two heads: policy (actions) and value
        self.action_head = nn.Linear(128, self.action_size)
        self.value_head = nn.Linear(128, 1)

        self.to(device)

    def forward(self, x):
        # Add channel dimension and convert to float
        x = x.view(-1, 1, self.rows, self.cols)

        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(-1, self.flat_size)

        # Fully connected layer
        x = F.relu(self.fc1(x))

        # Policy head (action probabilities)
        action_logits = self.action_head(x)

        # Value head (game outcome prediction)
        value_logit = self.value_head(x)

        return action_logits, value_logit

    def predict(self, boards: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        """
        Input: list of boards as numpy arrays of shape (rows, cols)
        Returns: probability distributions over actions and value estimates
        """
        # Convert list of boards to tensor
        boards_tensor = torch.FloatTensor(np.array(boards)).to(self.device)
        if boards_tensor.dim() == 2:  # Single board
            boards_tensor = boards_tensor.view(1, self.rows, self.cols)
        elif boards_tensor.dim() == 3:  # Batch of boards
            boards_tensor = boards_tensor.view(-1, self.rows, self.cols)

        self.eval()
        with torch.no_grad():
            action_logits, value_logit = self.forward(boards_tensor)
            action_probs = F.softmax(action_logits, dim=1)
            value = torch.tanh(value_logit)

        # Convert to list of numpy arrays
        action_probs_list = list(action_probs.cpu().numpy())
        value_list = list(value.cpu().numpy().flatten())

        return action_probs_list, value_list
