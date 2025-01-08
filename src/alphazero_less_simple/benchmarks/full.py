import numpy as np

from alphazero_simple.base_game import BaseGame
from alphazero_simple.base_model import BaseModel
from alphazero_simple.monte_carlo_tree_search import MCTS


class FullBenchmark:
    def __init__(
        self,
        game: BaseGame,
        model: BaseModel,
        num_simulations: int,
        temperature=1.0,
    ):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature

    def run(self, board: np.ndarray, to_play: int) -> int:
        board = self.game.get_canonical_board(board, to_play)

        mcts = MCTS(
            game=self.game,
            model=self.model,
            num_simulations=self.num_simulations,
        )

        root = mcts.run(board, to_play)

        action_probs = [0 for _ in range(self.game.get_action_size())]
        for k, v in root.children.items():
            action_probs[k] = v.visit_count
        action_probs = action_probs / np.sum(action_probs)

        action = root.select_action(temperature=1)

        return action
