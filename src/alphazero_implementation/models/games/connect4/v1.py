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
