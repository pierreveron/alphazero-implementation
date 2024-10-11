import torch
from simulator.game.connect import State  # type: ignore[attr-defined]
from torch import Tensor, nn

from alphazero_implementation.models.games.connect4.connect4_model import Connect4Model


class BasicNN(Connect4Model):
    def __init__(self, height: int, width: int, max_actions: int, num_players: int):
        super(BasicNN, self).__init__()  # type: ignore[call-arg]
        self.flatten = nn.Flatten()
        self.shared_layers = nn.Sequential(
            nn.Linear(height * width, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(512, max_actions)

        # Value head
        self.value_head = nn.Linear(512, num_players)

        self.learning_rate = 1e-3

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
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

    # def _states_to_tensor(self, states: list[State]) -> Tensor:
    #     batch_size = len(states)
    #     first_state = states[0]
    #     height, width = first_state.config.height, first_state.config.width

    #     # Initialize tensors for all states
    #     inputs = torch.zeros((batch_size, 3, height, width))

    #     for i, state in enumerate(states):
    #         tensor = torch.tensor(state.grid)  # type: ignore[arg-type]

    #         # Create the three channels
    #         available_moves = (tensor[0] == -1).float()  # Top row for available moves
    #         current_player = (tensor == state.player).float()
    #         opponent = (tensor == (1 - state.player)).float()

    #         # Stack the channels for this state
    #         inputs[i] = torch.stack([available_moves, current_player, opponent])

    #     return inputs
