from alphazero_implementation.games.game import Game
from alphazero_implementation.games.state import Action, GameState


class Bounce(Game):
    def get_reward(self) -> float:
        return 0

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (-1, 6, 9, 6)

    @property
    def initial_state(self) -> GameState:
        raise NotImplementedError("Bounce state not implemented")

    @property
    def num_actions(self) -> int:
        raise NotImplementedError("Bounce num_actions not implemented")

    def is_terminal(self, state: GameState) -> bool:
        raise NotImplementedError("Bounce is_terminal not implemented")

    def current_player(self, state: GameState) -> int:
        raise NotImplementedError("Bounce current_player not implemented")

    def next_state(self, state: GameState, action: Action) -> GameState:
        raise NotImplementedError("Bounce next_state not implemented")

    def display(self, state: GameState) -> None:
        raise NotImplementedError("Bounce display not implemented")

    def winner(self, state: GameState) -> int:
        raise NotImplementedError("Bounce winner not implemented")
