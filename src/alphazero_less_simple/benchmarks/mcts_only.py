import numpy as np

from alphazero_simple.base_game import BaseGame
from alphazero_simple.monte_carlo_tree_search import MCTS


class MCTSOnlyBenchmark:
    def __init__(self, game: BaseGame, num_simulations: int, temperature: float):
        self.game = game
        self.mcts = MCTS(game, model=None, num_simulations=num_simulations)
        self.temperature = temperature

    def run(self, board: np.ndarray, to_play: int):
        board = self.game.get_canonical_board(board, to_play)

        root = self.mcts.run_single(board, to_play)

        action = root.select_action(self.temperature)

        return action
