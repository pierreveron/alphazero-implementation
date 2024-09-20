from alphazero_implementation.games.game import Game
from alphazero_implementation.games.state import Action, GameState


class ChineseCheckers(Game):
    def get_reward(self) -> float:
        return 0

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (-1, 5, 17, 17)

    @property
    def initial_state(self) -> GameState:
        raise NotImplementedError("ChineseCheckers state not implemented")

    @property
    def num_actions(self) -> int:
        raise NotImplementedError("ChineseCheckers num_actions not implemented")

    def is_terminal(self, state: GameState) -> bool:
        raise NotImplementedError("ChineseCheckers is_terminal not implemented")

    def current_player(self, state: GameState) -> int:
        raise NotImplementedError("ChineseCheckers current_player not implemented")

    def next_state(self, state: GameState, action: Action) -> GameState:
        raise NotImplementedError("ChineseCheckers next_state not implemented")

    def display(self, state: GameState) -> None:
        raise NotImplementedError("ChineseCheckers display not implemented")

    def winner(self, state: GameState) -> int:
        raise NotImplementedError("ChineseCheckers winner not implemented")
