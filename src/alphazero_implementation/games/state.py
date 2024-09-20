from typing import Protocol


class Action:
    def __init__(self):
        pass


class GameState(Protocol):
    legal_actions: set[Action]
    last_action: Action | None

    def __init__(self):
        self.legal_actions = set()
        self.last_action = None

    def is_terminal(self) -> bool: ...

    def reward(self) -> float: ...

    def play(self, action: Action) -> "GameState": ...
