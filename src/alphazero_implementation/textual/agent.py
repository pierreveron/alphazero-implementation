import random
from abc import ABC, abstractmethod

from simulator.game.connect import Action, State

from alphazero_implementation.mcts.agent import MCTSAgent
from alphazero_implementation.mcts.mcgs import Node
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

    def __init__(self, model: Model) -> None:
        self.model = model

    def predict_best_action(self, state: State) -> Action:
        # return self.model.predict([state])[0][0]
        agent = MCTSAgent(
            self.model,
            num_episodes=100,
            simulations_per_episode=100,
            initial_state=state,
        )
        policy = agent.compute_policy(Node(state), {})
        action = max(policy.items(), key=lambda x: x[1])[0]
        return action
