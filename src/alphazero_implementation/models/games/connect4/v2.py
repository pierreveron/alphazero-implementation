import torch
from simulator.game.connect import State  # type: ignore[attr-defined]
from torch import Tensor, nn

from alphazero_implementation.models.games.connect4.connect4_model import Connect4Model


class CNNModel(Connect4Model):
    def __init__(self, height: int, width: int, max_actions: int, num_players: int):
        super().__init__(height, width, max_actions, num_players)

        # Paramètres des couches CNN
        self.channels = [
            3,
            64,
            128,
            256,
        ]  # Channels d'entrée et de sortie pour chaque couche

        # Couches CNN
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

        # Calcul de la taille après les couches CNN
        self.conv_output_size = self.channels[-1] * height * width

        # Couches fully connected
        self.shared_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Policy head
        self.policy_head = nn.Linear(512, max_actions)

        # Value head
        self.value_head = nn.Linear(512, num_players)

        self.learning_rate = 1e-3

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Move input tensor to the same device as the model
        x = x.to(next(self.parameters()).device)

        # Passage dans les couches CNN
        x = self.conv_layers(x)

        # Flatten pour les couches fully connected
        x = x.view(x.size(0), -1)

        # Couches partagées
        shared_output = self.shared_layers(x)

        # Têtes de sortie
        policy_logits = self.policy_head(shared_output)
        value = self.value_head(shared_output)

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
            available_moves = torch.zeros(height, width)
            available_moves[0] = (
                tensor[0] == -1
            ).float()  # Top row for available moves
            current_player = (tensor == state.player).float()
            opponent = (tensor == (1 - state.player)).float()

            # Stack the channels for this state
            inputs[i] = torch.stack([available_moves, current_player, opponent])

        return inputs
