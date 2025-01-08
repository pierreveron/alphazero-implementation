import numpy as np

from alphazero_simple.base_game import BaseGame
from alphazero_simple.base_model import BaseModel


class NetworkOnlyBenchmark:
    def __init__(self, game: BaseGame, model: BaseModel, temperature=1.0):
        self.game = game
        self.model = model
        self.temperature = temperature

    def run(self, board: np.ndarray, to_play: int) -> int:
        board = self.game.get_canonical_board(board, to_play)

        valid_moves = self.game.get_valid_moves(board)

        if self.temperature == float("inf"):
            valid_indices = np.where(valid_moves)[0]
            return int(np.random.choice(valid_indices))

        [policy], _ = self.model.predict([board])

        valid_moves = self.game.get_valid_moves(board)
        policy = policy * valid_moves  # mask invalid moves
        policy /= np.sum(policy)

        if self.temperature == 0:
            return int(np.argmax(policy))
        else:
            # Apply temperature by exponentiating probabilities
            probs = policy ** (1 / self.temperature)
            # Renormalize
            probs = probs / np.sum(probs)
            # Choose action based on probabilities
            return int(np.random.choice(len(policy), p=probs))
