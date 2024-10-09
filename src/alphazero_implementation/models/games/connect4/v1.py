from typing import Any

import lightning as L
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam  # type: ignore[import]


class BasicNN(L.LightningModule):
    def __init__(self):
        super(BasicNN, self).__init__()  # type: ignore[call-arg]
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

        self.learning_rate = 1e-3

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
