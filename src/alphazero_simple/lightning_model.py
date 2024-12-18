import lightning as L
import torch
from torch import nn


class AlphaZeroLitModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-4,
    ):
        """
        PyTorch Lightning module for AlphaZero training.

        Args:
            model: The neural network model to train
            learning_rate: Learning rate for the optimizer
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x)

    def loss_pi(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the policy loss using cross-entropy.

        Args:
            targets: Target policy probabilities
            outputs: Predicted policy probabilities
        """
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the value loss using MSE.

        Args:
            targets: Target values
            outputs: Predicted values
        """
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Training step for Lightning.

        Args:
            batch: Tuple of (boards, policy targets, value targets)
            batch_idx: Index of the current batch
        """
        boards, target_pis, target_vs = batch

        # Get model predictions
        out_pi, out_v = self(boards)

        # Calculate losses
        loss_pi = self.loss_pi(target_pis, out_pi)
        loss_v = self.loss_v(target_vs, out_v)
        total_loss = loss_pi + loss_v

        # Log metrics
        self.log("train_loss_pi", loss_pi, prog_bar=True)
        self.log("train_loss_v", loss_v, prog_bar=True)
        self.log("train_loss", total_loss, prog_bar=True)

        return {"loss": total_loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:  # type: ignore[attr-defined]
        """Configure the Adam optimizer."""
        return torch.optim.Adam(  # type: ignore[attr-defined]
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
