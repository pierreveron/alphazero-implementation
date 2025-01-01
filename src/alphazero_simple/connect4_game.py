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
        # Pre-compute all possible win pattern directions
        self.directions = [
            (0, 1),  # horizontal
            (1, 0),  # vertical
            (1, 1),  # diagonal positive
            (1, -1),  # diagonal negative
        ]
        # Create convolution kernels for each direction
        self._init_win_kernels()

    def _init_win_kernels(self):
        """Initialize convolution kernels for win checking"""
        self.win_kernels = []
        for dr, dc in self.directions:
            kernel = np.zeros(
                (self.win_length * abs(dr) or 1, self.win_length * abs(dc) or 1)
            )
            for i in range(self.win_length):
                r = i * dr if dr else 0
                c = i * dc if dc else 0
                kernel[r, c] = 1
            self.win_kernels.append(kernel)

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
        """Checks for 4 in a row using convolution"""
        # Create player-specific board
        player_board = (board == player).astype(np.int8)

        # Check each direction using convolution
        for kernel in self.win_kernels:
            # Convolve the board with the kernel
            conv = np.correlate(player_board.ravel(), kernel.ravel(), mode="valid")
            if (conv >= self.win_length).any():
                return True
        return False

    def get_reward_for_player(self, board: np.ndarray, player: int) -> float | None:
        """Returns: None if game not ended, 1 if player won, -1 if player lost, 0 if draw"""
        # Check current player first (most common case)
        if self.is_win(board, player):
            return 1.0
        # Only check opponent if current player hasn't won
        if self.is_win(board, -player):
            return -1.0
        # Only check for moves if no one has won
        if self.has_legal_moves(board):
            return None
        return 0.0

    def get_canonical_board(self, board: np.ndarray, player: int) -> np.ndarray:
        return player * board
