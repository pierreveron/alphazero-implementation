import copy
from abc import ABC, abstractmethod

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor
from torch.optim import Adam  # type: ignore[import]
from torch.utils.data import TensorDataset


class BaseModel(ABC, L.LightningModule):
    """
    An abstract class for a model that can be used in the MCTS simulation.
    """

    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()  # This line saves all __init__ arguments as hyperparameters
        self.learning_rate = learning_rate
        self.model_name = self.__class__.__name__
        self.save_hyperparameters({"model_name": self.model_name})

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, policy_target, value_target = batch

        # Get model predictions
        policy_logits, value_logits = self.forward(x)

        # Calculate losses
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        value_loss = F.mse_loss(value_logits, value_target)
        total_loss = policy_loss + value_loss

        self.log("train_loss", total_loss)
        self.log("policy_loss", policy_loss)
        self.log("value_loss", value_loss)

        return total_loss

    def configure_optimizers(self):
        # Deepmind's AlphaZero paper uses RMSProp with a learning rate of 0.001
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    # def predict(self, states: list[State]) -> tuple[list[ActionPolicy], list[Value]]:
    def predict(self, boards: list[NDArray]) -> tuple[list[NDArray], list[float]]:
        """
        Base prediction method that should be implemented by subclasses
        """
        pass

    def format_dataset(
        self, states: list[NDArray], policies: list[NDArray], values: list[float]
    ) -> TensorDataset:
        state_inputs = self._states_to_tensor(states)
        policy_targets = torch.FloatTensor(np.array(policies))
        value_targets = torch.FloatTensor(np.array(values).astype(np.float64))
        return TensorDataset(state_inputs, policy_targets, value_targets)

    @abstractmethod
    def _states_to_tensor(self, boards: list[np.ndarray]) -> Tensor:
        pass

    # @abstractmethod
    # def _policies_to_tensor(self, policies: list[NDArray]) -> Tensor:
    #     pass

    def get_inference_clone(self):
        """Create a clone of the model for inference"""
        clone = copy.deepcopy(self)
        clone.eval()  # Set to evaluation mode
        return clone
