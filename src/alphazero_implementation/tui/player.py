import random
from abc import ABC, abstractmethod

from simulator.game.connect import Action, State  # type: ignore[attr-defined]

from alphazero_implementation.core.search.mcts import AlphaZeroSearch, Node
from alphazero_implementation.models.model import Model


class Player(ABC):
    """Abstract base class of AI player."""

    @abstractmethod
    def play(self, state: State) -> Action:
        pass


class AlphaZeroPlayer(Player):
    """AlphaZero agent that uses MCTS with neural network guidance to select moves.

    This agent implements the AlphaZero algorithm, using Monte Carlo Tree Search (MCTS)
    guided by a neural network to select actions. The agent can operate with different
    temperature settings to control exploration vs exploitation.

    Args:
        model (Model): Neural network model that provides policy and value predictions
        mcts_simulation (int, optional): Number of MCTS simulations to run. Defaults to 100.
        temperature (float, optional): Temperature parameter for action selection:
            - temperature = 0: Deterministic selection (highest probability move)
            - 0 < temperature < inf: Stochastic selection weighted by probabilities
            - temperature = inf: Uniform random selection
            Defaults to 1.0.
    """

    def __init__(
        self, model: Model, *, mcts_simulation: int = 100, temperature: float = 1.0
    ) -> None:
        self.model = model
        self.mcts_simulation = mcts_simulation
        self.temperature = temperature

    def play(self, state: State) -> Action:
        """Predict the best action for the given game state.

        Uses MCTS with neural network guidance to search for the best move. The search
        process is controlled by the temperature parameter:
        - At temperature=0, selects the move with highest visit count
        - At 0<temperature<inf, samples moves proportional to their visit counts
        - At temperature=inf, selects a random legal move

        Args:
            state (State): Current game state

        Returns:
            Action: Selected action to take
        """
        if self.temperature == float("inf"):
            return random.choice(state.actions)

        agent = AlphaZeroSearch(
            model=self.model,
            num_simulations=self.mcts_simulation,
        )
        policy = agent.run(Node(state))

        if self.temperature == 0:
            action = max(policy.items(), key=lambda x: x[1])[0]
        else:
            # Apply temperature by exponentiating probabilities
            probs = [p ** (1 / self.temperature) for p in policy.values()]
            # Renormalize
            total = sum(probs)
            probs = [p / total for p in probs]
            action = random.choices(list(policy.keys()), weights=probs)[0]

        return action
