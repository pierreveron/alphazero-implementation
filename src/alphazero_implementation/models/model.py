from abc import ABC, abstractmethod

import lightning as L
import torch.nn.functional as F
from simulator.game.connect import Action, State  # type: ignore[import]
from torch import Tensor
from torch.optim import Adam  # type: ignore[import]

# ActionPolicy represents a probability distribution over available actions in a given state.
# It maps each possible action to its probability of being selected, providing a strategy
# for action selection based on the current game state.
ActionPolicy = dict[Action, float]


# Value represents the estimated value of a game state for each player.
# It is a list of floating-point numbers, where each element corresponds
# to the expected outcome or utility for a specific player in the current game state.
# The list's length matches the number of players in the game.
Value = list[float]


class Model(ABC, L.LightningModule):
    """
    An abstract class for a model that can be used in the MCTS simulation.
    """

    learning_rate: float = 1e-3

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
    def _states_to_tensor(self, states: list[State]) -> Tensor:
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
