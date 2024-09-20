from abc import ABC, abstractmethod

from alphazero_implementation.games.state import Action, GameState


class Model(ABC):
    """
    An abstract class for a model that can be used in the MCTS simulation.
    """

    @abstractmethod
    def predict(self, state: GameState) -> Action:
        """
        Predicts the next action to take in the game given the current state.
        """
        pass
