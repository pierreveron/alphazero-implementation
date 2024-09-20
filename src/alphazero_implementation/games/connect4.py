from alphazero_implementation.games.game import Game


class Connect4(Game):
    def get_reward(self) -> float:
        return 0

    def get_input_shape(self) -> tuple[int, int, int, int]:
        return (-1, 3, 6, 7)
