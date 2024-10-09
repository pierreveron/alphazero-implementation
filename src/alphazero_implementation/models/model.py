from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from simulator.game.connect import Action, State  # type: ignore[import]

# ActionPolicy represents a probability distribution over available actions in a given state.
# It maps each possible action to its probability of being selected, providing a strategy
# for action selection based on the current game state.
ActionPolicy = dict[Action, float]


# Value represents the estimated value of a game state for each player.
# It is a numpy array of floating-point numbers, where each element corresponds
# to the expected outcome or utility for a specific player in the current game state.
# The array's length matches the number of players in the game.
Value = NDArray[np.float64]


class Model(ABC):
    """
    An abstract class for a model that can be used in the MCTS simulation.
    """

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
