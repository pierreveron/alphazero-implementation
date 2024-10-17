import random
import time
from abc import ABC, abstractmethod

from simulator.game.connect import Action, State

from alphazero_implementation.models.model import Model


class Agent(ABC):
    """Abstract base class of AI player."""

    @abstractmethod
    def predict(self, state: State) -> dict[Action, float]:
        pass


class RandomAgent(Agent):
    """Uniform distribution."""

    def predict(self, state: State) -> dict[Action, float]:
        time.sleep(random.random())
        actions = state.actions
        return {action: 1 / len(actions) for action in actions}


class AlphaZeroAgent(Agent):
    """AlphaZero agent."""

    def __init__(self, model: Model) -> None:
        self.model = model

    def predict(self, state: State) -> dict[Action, float]:
        policies, _ = self.model.predict([state])
        return policies[0]
