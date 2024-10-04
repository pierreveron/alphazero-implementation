from typing import Iterator

import torch
from torch import Tensor, nn


class AlphaZeroLoss(nn.Module):
    def __init__(self, c: float = 1.0):
        super(AlphaZeroLoss, self).__init__()  # type: ignore[call-arg]
        self.c = c  # Regularization coefficient

    def forward(
        self,
        v: Tensor,
        z: Tensor,
        pi: Tensor,
        p: Tensor,
        theta: Tensor | Iterator[Tensor],
    ) -> Tensor:
        """
        Parameters:
        v: predicted value
        z: target value
        pi: search probabilities (π)
        p: policy vector
        theta: model parameters
        """
        # Mean squared error term: (z - v)^2
        mse_loss: Tensor = (z - v) ** 2

        # Cross-entropy term: -π^T log p
        # Adding a small epsilon to prevent log(0)
        # eps = 1e-8
        # cross_entropy = -torch.sum(pi * torch.log(p + eps))
        # Cross-entropy loss (second term)
        cross_entropy: Tensor = -torch.dot(pi, torch.log(p))

        # L2 regularization term: c||θ||^2
        if isinstance(theta, Tensor):
            l2_reg: Tensor = self.c * torch.sum(theta**2)
        else:
            c_tensor = torch.tensor(self.c, dtype=torch.float32)
            l2_reg: Tensor = c_tensor * sum(torch.sum(param**2) for param in theta)

        # Total loss
        total_loss: Tensor = mse_loss + cross_entropy + l2_reg

        return total_loss
