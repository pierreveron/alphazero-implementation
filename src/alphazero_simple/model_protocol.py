from typing import Protocol

from numpy.typing import NDArray


class GameModel(Protocol):
    def predict(self, board: NDArray) -> tuple[NDArray, float]:
        """
        Predict action probabilities and value for a given board state.

        Args:
            board: The game board state as a numpy array

        Returns:
            tuple containing:
            - NDArray: probability distribution over actions
            - float: value estimate of the position
        """
        ...
