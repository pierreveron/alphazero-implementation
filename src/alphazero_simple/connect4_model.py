import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base_model import BaseModel


class Connect4Model(BaseModel):
    def __init__(
        self,
        board_size: tuple[int, int],
        action_size: int,
    ):
        super(Connect4Model, self).__init__()  # type: ignore[no-untyped-call]

        self.rows, self.cols = board_size
        self.action_size = action_size

        # Define channel sizes
        self.channels = [1, 64, 128, 256]  # Starting with 1 channel for Connect4

        # CNN layers with BatchNorm
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(),
            nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(),
        )

        # Calculate flattened size
        self.flat_size = self.channels[-1] * self.rows * self.cols

        # Shared layers with dropout
        self.shared_layers = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Policy and value heads
        self.action_head = nn.Linear(512, self.action_size)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # Add channel dimension and convert to float
        x = x.view(-1, 1, self.rows, self.cols)

        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(-1, self.flat_size)

        # Shared layers
        x = self.shared_layers(x)

        # Policy head (action probabilities)
        action_logits = self.action_head(x)

        # Value head (game outcome prediction)
        value_logit = self.value_head(x)

        return action_logits, value_logit.view(-1)

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
