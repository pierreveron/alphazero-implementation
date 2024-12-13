from abc import ABC, abstractmethod

import numpy as np


class BaseGame(ABC):
    """
    Abstract base class for board games.
    Defines the interface that all game implementations must follow.
    """

    @abstractmethod
    def get_init_board(self) -> np.ndarray:
        """Returns the initial board state"""
        pass

    @abstractmethod
    def get_board_size(self) -> int | tuple[int, int]:
        """Returns the size of the board"""
        pass

    @abstractmethod
    def get_action_size(self) -> int:
        """Returns the number of possible actions"""
        pass

    @abstractmethod
    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> tuple[np.ndarray, int]:
        """Returns (next_state, next_player) tuple"""
        pass

    @abstractmethod
    def has_legal_moves(self, board: np.ndarray) -> bool:
        """Returns True if there are legal moves available"""
        pass

    @abstractmethod
    def get_valid_moves(self, board: np.ndarray) -> list[int]:
        """Returns a list of valid moves"""
        pass

    @abstractmethod
    def is_win(self, board: np.ndarray, player: int) -> bool:
        """Returns True if player has won the game"""
        pass

    @abstractmethod
    def get_reward_for_player(self, board: np.ndarray, player: int) -> float | None:
        """
        Returns:
            None if game has not ended
            1 if player won
            -1 if player lost
            0 if game ended in draw
        """
        pass

    def get_canonical_board(self, board: np.ndarray, player: int) -> np.ndarray:
        """Returns the canonical form of board from given player's perspective"""
        return player * board
