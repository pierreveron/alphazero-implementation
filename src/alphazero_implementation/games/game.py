from abc import ABC, abstractmethod


class Game(ABC):
    def __init__(self):
        self.state = None

    @abstractmethod
    def get_reward(self) -> float:
        """
        Returns the reward for the current state of the game.
        """
        pass

    @abstractmethod
    def get_input_shape(self) -> tuple[int, int, int, int]:
        """
        Returns the shape of the input tensor for the neural network.
        """
        pass
