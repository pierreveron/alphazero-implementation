from dataclasses import dataclass, field
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        """Convert Sample to dictionary for JSON serialization"""
        return {
            "state": self.state.to_json(),
            "policy": {
                str(action.to_json()): prob for action, prob in self.policy.items()
            },
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sample":
        """Create Sample from dictionary"""
        return cls(
            state=State.from_json(data["state"]),
            policy={
                Action.from_json(eval(action_data)): prob
                for action_data, prob in data["policy"].items()
            },
            value=data["value"],
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Convert Episode to dictionary for JSON serialization"""
        return {"samples": [sample.to_dict() for sample in self.samples]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        """Create Episode from dictionary"""
        episode = cls()
        episode.samples = [
            Sample.from_dict(sample_data) for sample_data in data["samples"]
        ]
        return episode


@dataclass
class Iteration:
    """Represents a complete training iteration"""

    episodes: list[Episode]
    loss: float
    policy_loss: float
    value_loss: float
    accuracy: float
