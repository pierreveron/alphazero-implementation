from typing import Protocol

import numpy as np
import tqdm

from alphazero_simple.base_game import BaseGame


class Player(Protocol):
    def run(self, board: np.ndarray, to_play: int) -> int:
        """Run the player on a given board state.

        Args:
            board: The current board state
            to_play: The current player (1 or -1)

        Returns:
            The selected action/move
        """
        ...


class Arena:
    def __init__(self, game: BaseGame):
        self.game = game

    def play_game(self, player1: Player, player2: Player) -> float:
        """Play a full game between two players and return the winner.

        Args:
            player1: First player (can be FullBenchmark or MCTSOnlyBenchmark)
            player2: Second player (can be FullBenchmark or MCTSOnlyBenchmark)

        Returns:
            winner: 1 for player1 win, -1 for player2 win, 0 for draw
        """
        board = self.game.get_init_board()
        current_player = 1

        while True:
            # Get current player's move
            if current_player == 1:
                action = player1.run(board, current_player)
            else:
                action = player2.run(board, current_player)

            board, current_player = self.game.get_next_state(
                board, current_player, action
            )

            reward = self.game.get_reward_for_player(board, 1)

            if reward is not None:
                return reward

    def play_matches(
        self,
        player1: Player,
        player2: Player,
        num_games: int,
        progress_bar_position: int = 0,
    ) -> tuple[int, int, int]:
        """Play multiple games between two players and return the win statistics.

        Args:
            player1: First player
            player2: Second player
            num_games: Number of games to play
            position: Position of the progress bar (for nested bars)

        Returns:
            Tuple of (player1_wins, player2_wins, draws)
        """
        player1_wins = 0
        player2_wins = 0
        draws = 0

        # Configure tqdm for nested bars
        for i in tqdm.notebook.trange(
            num_games,
            desc=f"Playing games ({player1.__class__.__name__} vs {player2.__class__.__name__})",
            position=progress_bar_position,
            leave=progress_bar_position == 0,
        ):
            # Alternate who plays first
            if i % 2 == 0:
                winner = self.play_game(player1, player2)
            else:
                winner = self.play_game(player2, player1)
                # Flip the winner since players were swapped
                winner = -winner

            if winner == 1:
                player1_wins += 1
            elif winner == -1:
                player2_wins += 1
            else:
                draws += 1

        return player1_wins, player2_wins, draws
