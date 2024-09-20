import random

from alphazero_implementation.games.state import Action, GameState
from alphazero_implementation.models.model import Model


class RandomModel(Model):
    """
    A model that returns a uniformly distributed random action to use in the MCTS simulation.
    """

    def __init__(self):
        pass

    def predict(self, state: GameState) -> Action:
        return random.choice(list(state.legal_actions))
