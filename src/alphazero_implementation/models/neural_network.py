from alphazero_implementation.games.state import Action, GameState
from alphazero_implementation.models.model import Model


class NeuralNetwork(Model):
    """
    A model that uses a neural network to predict the action to take.
    """

    def __init__(self):
        pass

    def predict(self, state: GameState) -> Action:
        return Action()
