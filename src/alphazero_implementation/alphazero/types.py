from dataclasses import dataclass

from simulator.game.connect import Action, State  # type: ignore[import]


@dataclass
class Sample:
    """Represents a single training sample from self-play"""

    state: State
    policy: dict[Action, float]  # Maps actions to their MCTS visit probabilities
    value: list[float]  # The actual game outcome from this state


@dataclass
class Episode:
    """Represents a complete self-play game episode"""

    samples: list[Sample]


@dataclass
class Iteration:
    """Represents a complete training iteration"""

    episodes: list[Episode]
    loss: float
    policy_loss: float
    value_loss: float
    accuracy: float
