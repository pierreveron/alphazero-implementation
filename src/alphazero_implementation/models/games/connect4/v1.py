from typing import Any

import lightning as L
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam  # type: ignore[import]


class BasicNN(L.LightningModule):
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

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        x, policy_target, value_target = batch
        policy_logits, value = self(x)

        # Calculate policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_logits, policy_target)

        # Calculate value loss (mean squared error)
        value_loss = F.mse_loss(value, value_target)

        # Combine losses (you can adjust the weighting if needed)
        total_loss = policy_loss + value_loss

        self.log("train_loss", total_loss)  # type: ignore[arg-type]
        self.log("policy_loss", policy_loss)  # type: ignore[arg-type]
        self.log("value_loss", value_loss)  # type: ignore[arg-type]

        return total_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
