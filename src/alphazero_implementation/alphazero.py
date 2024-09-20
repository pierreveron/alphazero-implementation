from .games.game import Game
from .games.state import Action, GameState
from .mcts.mcts import MCTS
from .models.neural_network import NeuralNetwork


class AlphaZero:
    def __init__(self, game: Game, model: NeuralNetwork):
        self.game = game
        self.model = model
        self.mcts = MCTS(simulation_model=model)

    def self_play(self, num_games: int) -> list[tuple[GameState, Action]]:
        # Implement self-play logic
        return []

    def train(self, training_data: list[tuple[GameState, Action]]):
        # Implement neural network training
        return

    def evaluate(self, opponent: NeuralNetwork) -> bool:
        # Implement model evaluation
        return False

    def run(self, num_iterations: int):
        for _ in range(num_iterations):
            game_data = self.self_play(num_games=100)
            self.train(game_data)
            if self.evaluate(opponent=self.model):
                self.model.save(filepath="model.pt")
