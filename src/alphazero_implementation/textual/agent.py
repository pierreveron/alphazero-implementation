import random
from abc import ABC, abstractmethod

from simulator.game.connect import Action, State

from alphazero_implementation.mcts_v2.mcts import AlphaZeroMCTS
from alphazero_implementation.mcts_v2.node import Node
from alphazero_implementation.models.model import Model


class Agent(ABC):
    """Abstract base class of AI player."""

    @abstractmethod
    def predict_best_action(self, state: State) -> Action:
        pass


class RandomAgent(Agent):
    """Uniform distribution."""

    def predict_best_action(self, state: State) -> Action:
        return random.choice(state.actions)


class AlphaZeroAgent(Agent):
    """AlphaZero agent."""

    def __init__(self, model: Model, *, stochastic: bool = False) -> None:
        self.model = model
        self.stochastic = stochastic

    def predict_best_action(self, state: State) -> Action:
        # policy = self.model.predict([state])[0][0]
        # action = max(policy.items(), key=lambda x: x[1])[0]
        # return action

        agent = AlphaZeroMCTS(self.model)
        policy = agent.run(Node(state), 100)

        if self.stochastic:
            action = random.choices(list(policy.keys()), weights=list(policy.values()))[
                0
            ]
        else:
            action = max(policy.items(), key=lambda x: x[1])[0]

        return action
