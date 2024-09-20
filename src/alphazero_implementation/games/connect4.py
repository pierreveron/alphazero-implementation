from alphazero_implementation.games.game import Game
from alphazero_implementation.games.state import Action, GameState


class Connect4(Game):
    def get_reward(self) -> float:
        return 0

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (-1, 3, 6, 7)

    @property
    def initial_state(self) -> GameState:
        raise NotImplementedError("Connect4 state not implemented")

    @property
    def num_actions(self) -> int:
        raise NotImplementedError("Connect4 num_actions not implemented")

    def is_terminal(self, state: GameState) -> bool:
        raise NotImplementedError("Connect4 is_terminal not implemented")

    def current_player(self, state: GameState) -> int:
        raise NotImplementedError("Connect4 current_player not implemented")

    def next_state(self, state: GameState, action: Action) -> GameState:
        raise NotImplementedError("Connect4 next_state not implemented")

    def display(self, state: GameState) -> None:
        raise NotImplementedError("Connect4 display not implemented")

    def winner(self, state: GameState) -> int:
        raise NotImplementedError("Connect4 winner not implemented")
