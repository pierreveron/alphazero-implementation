from abc import ABC, abstractmethod

from alphazero_implementation.games.state import Action, GameState


class Game(ABC):
    @abstractmethod
    def get_reward(self) -> float:
        """
        Returns the reward for the current state of the game.
        """
        pass

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, int, int, int]:
        """
        Returns the shape of the input tensor for the neural network.
        """
        pass

    @property
    @abstractmethod
    def initial_state(self) -> GameState:
        """
        Returns the initial state of the game.
        """
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """
        Returns the number of possible actions in the game.
        """
        pass

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """
        Returns whether the game is in a terminal state.
        """
        pass

    @abstractmethod
    def current_player(self, state: GameState) -> int:
        """
        Returns the current player of the game.
        """
        pass

    @abstractmethod
    def next_state(self, state: GameState, action: Action) -> GameState:
        """
        Returns the next state of the game.
        """
        pass

    @abstractmethod
    def display(self, state: GameState) -> None:
        """
        Displays the current state of the game.
        """
        pass

    @abstractmethod
    def winner(self, state: GameState) -> int:
        """
        Returns the winner of the game.
        """
        pass
