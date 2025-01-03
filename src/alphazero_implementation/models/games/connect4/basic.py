import torch
from simulator.game.connect import State  # type: ignore[attr-defined]
from torch import Tensor, nn

from .model import Connect4Model


class BasicNN(Connect4Model):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.shared_layers = nn.Sequential(
            nn.Linear(self.board_height * self.board_width, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(512, self.board_width)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(512, 2),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Move input tensor to the same device as the model
        x = x.to(self.shared_layers[0].weight.device)

        x = self.flatten(x)
        shared_output = self.shared_layers(x)

        policy_logits = self.policy_head(shared_output)
        value = self.value_head(shared_output)

        return policy_logits, value

    def _states_to_tensor(self, states: list[State]) -> Tensor:
        import numpy as np
        from numpy.typing import NDArray

        grids: list[NDArray[np.float64]] = [state.grid for state in states]  # type: ignore[attr-defined]
        stacked = np.stack(grids)
        return torch.tensor(stacked, dtype=torch.float32)
