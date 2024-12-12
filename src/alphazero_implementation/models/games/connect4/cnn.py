import torch
from simulator.game.connect import State  # type: ignore[attr-defined]
from torch import Tensor, nn

from .model import Connect4Model


class CNNModel(Connect4Model):
    def __init__(self):
        super().__init__()

        # Parameters for CNN layers
        self.channels = [
            3,
            64,
            128,
            256,
        ]  # Input and output channels for each layer

        # CNN layers
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

        # Calculate size after CNN layers
        self.conv_output_size = self.channels[-1] * self.board_height * self.board_width

        # Fully connected layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Policy head
        self.policy_head = nn.Linear(512, self.board_width)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Tanh(),
        )

        self.learning_rate = 1e-3

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Move input tensor to the same device as the model
        x = x.to(next(self.parameters()).device)

        # Pass through CNN layers
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Shared layers
        shared_output = self.shared_layers(x)

        # Output heads
        policy_logits = self.policy_head(shared_output)
        value = self.value_head(shared_output)

        # Concatenate value and its negative along dim=1
        # Result shape: [batch_size, 2]
        value = torch.cat([value, -value], dim=1)

        return policy_logits, value

    def _states_to_tensor(self, states: list[State]) -> Tensor:
        batch_size = len(states)
        first_state = states[0]
        height, width = first_state.config.height, first_state.config.width

        # Initialize tensors for all states
        inputs = torch.zeros((batch_size, 3, height, width))

        for i, state in enumerate(states):
            tensor = torch.tensor(state.grid)  # type: ignore[arg-type]

            # Ensure tensor is 2D by reshaping if necessary
            # if len(tensor.shape) == 1:
            #     tensor = tensor.reshape(height, width)

            # Create the three channels
            available_moves = (tensor == -1).float()
            current_player = (tensor == state.player).float()
            opponent = (tensor == (1 - state.player)).float()

            # Stack the channels for this state
            inputs[i] = torch.stack([available_moves, current_player, opponent])

        return inputs
