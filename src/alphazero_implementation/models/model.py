import copy
from abc import ABC, abstractmethod

import lightning as L
import torch
import torch.nn.functional as F
from simulator.game.connect import State  # type: ignore[import]
from torch import Tensor
from torch.optim import Adam  # type: ignore[import]
from torch.utils.data import TensorDataset

from alphazero_implementation.models.types import ActionPolicy, Value


class Model(ABC, L.LightningModule):
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
        # Deepmind's AlphaZero paper uses RMSProp with a learning rate of 0.001
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
        return Adam(self.parameters(), lr=self.learning_rate)

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def predict(self, states: list[State]) -> tuple[list[ActionPolicy], list[Value]]:
        """
        Predict action probabilities and state values for a list of game states.

        This method takes a list of game states and returns two lists:
        1. A list of action policies: Each policy is a dictionary mapping legal actions to their probabilities.
        2. A list of state values: Each value is an array representing the estimated value of the state for each player.

        The action policies provide probability distributions over possible moves for each state,
        while the state values estimate the expected outcomes of the game from each state for each player.

        Args:
            states (list[State]): A list of game states to evaluate.

        Returns:
            tuple[list[ActionPolicy], list[Value]]: A tuple containing:
                - list[ActionPolicy]: A list of dictionaries, each mapping legal actions to their probabilities.
                - list[Value]: A list of arrays, each representing the estimated value of a state for each player.
        """
        pass

    def format_dataset(
        self, states: list[State], policies: list[ActionPolicy], values: list[Value]
    ) -> TensorDataset:
        state_inputs = self._states_to_tensor(states)
        policy_targets = self._policies_to_tensor(policies)
        value_targets = torch.FloatTensor(values)
        return TensorDataset(state_inputs, policy_targets, value_targets)

    @abstractmethod
    def _states_to_tensor(self, states: list[State]) -> Tensor:
        pass

    @abstractmethod
    def _policies_to_tensor(self, policies: list[ActionPolicy]) -> Tensor:
        pass

    def get_inference_clone(self):
        """Create a clone of the model for inference"""
        clone = copy.deepcopy(self)
        clone.eval()  # Set to evaluation mode
        return clone
