import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model_protocol import GameModel


class Connect4Model(nn.Module, GameModel):
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

        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)

    def predict(self, board):
        """
        Input: board in the form of numpy array
        Returns: probability distribution over actions and value estimate
        """
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.rows, self.cols)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
