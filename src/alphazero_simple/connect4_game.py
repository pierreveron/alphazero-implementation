import numpy as np

from .base_game import BaseGame


class Connect4Game(BaseGame):
    """
    Standard Connect4 game with:
        rows: 6
        columns: 7
        win_length: 4
    """

    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.win_length = 4

    def get_init_board(self) -> np.ndarray:
        return np.zeros((self.rows, self.columns), dtype=int)

    def get_board_size(self) -> tuple[int, int]:
        return (self.rows, self.columns)

    def get_action_size(self) -> int:
        return self.columns

    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> tuple[np.ndarray, int]:
        """Places a piece in the specified column and applies gravity"""
        b = np.copy(board)
        # Return the new game, but
        # change the perspective of the game with negative

        # Find the lowest empty row in the selected column
        for row in range(self.rows - 1, -1, -1):
            if b[row][action] == 0:
                b[row][action] = player
                break

        return (b, -player)

    def has_legal_moves(self, board: np.ndarray) -> bool:
        """Checks if there are any empty spaces in the top row"""
        return 0 in board[0]

    def get_valid_moves(self, board: np.ndarray) -> list[int]:
        """Returns a binary vector of valid moves (columns that aren't full)"""
        valid_moves = [0] * self.get_action_size()

        for col in range(self.columns):
            if board[0][col] == 0:  # If top cell is empty, move is valid
                valid_moves[col] = 1

        return valid_moves

    def is_win(self, board: np.ndarray, player: int) -> bool:
        """Checks for 4 in a row horizontally, vertically, or diagonally"""
        # Horizontal check
        for row in range(self.rows):
            for col in range(self.columns - self.win_length + 1):
                if all(board[row][col + i] == player for i in range(self.win_length)):
                    return True

        # Vertical check
        for row in range(self.rows - self.win_length + 1):
            for col in range(self.columns):
                if all(board[row + i][col] == player for i in range(self.win_length)):
                    return True

        # Diagonal check (positive slope)
        for row in range(self.rows - self.win_length + 1):
            for col in range(self.columns - self.win_length + 1):
                if all(
                    board[row + i][col + i] == player for i in range(self.win_length)
                ):
                    return True

        # Diagonal check (negative slope)
        for row in range(self.win_length - 1, self.rows):
            for col in range(self.columns - self.win_length + 1):
                if all(
                    board[row - i][col + i] == player for i in range(self.win_length)
                ):
                    return True

        return False

    def get_reward_for_player(self, board: np.ndarray, player: int) -> float | None:
        """Returns: None if game not ended, 1 if player won, -1 if player lost, 0 if draw"""
        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None
        return 0

    def get_canonical_board(self, board: np.ndarray, player: int) -> np.ndarray:
        return player * board
