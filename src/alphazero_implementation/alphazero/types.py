from dataclasses import dataclass, field

from simulator.game.connect import Action, State  # type: ignore[import]

# ActionPolicy represents a probability distribution over available actions in a given state.
# It maps each possible action to its probability of being selected, providing a strategy
# for action selection based on the current game state.
ActionPolicy = dict[Action, float]


# Value represents the estimated value of a game state for each player.
# It is a list of floating-point numbers, where each element corresponds
# to the expected outcome or utility for a specific player in the current game state.
# The list's length matches the number of players in the game.
Value = list[float]


@dataclass
class Sample:
    """Represents a single training sample from self-play"""

    state: State
    policy: ActionPolicy  # Maps actions to their MCTS visit probabilities
    value: Value  # The actual game outcome from this state


@dataclass
class Episode:
    """Represents a complete self-play game episode"""

    samples: list[Sample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def add_sample(self, sample: Sample) -> None:
        self.samples.append(sample)

    @property
    def current_state(self) -> State:
        return self.samples[-1].state


@dataclass
class Iteration:
    """Represents a complete training iteration"""

    episodes: list[Episode]
    loss: float
    policy_loss: float
    value_loss: float
    accuracy: float
