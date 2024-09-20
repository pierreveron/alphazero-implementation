from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Action:
    def __init__(self, index: int):
        self.index = index

    def __hash__(self) -> int:
        return hash(self.index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Action) and self.index == other.index


class GameState(ABC):
    def __init__(self):
        self.legal_actions: set[Action] = set()
        self.last_action: Action | None = None

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def reward(self) -> float:
        pass

    @abstractmethod
    def play(self, action: Action) -> "GameState":
        pass

    @abstractmethod
    def to_input(self) -> NDArray[np.float32]:
        pass
