from alphazero_implementation.games.game import Game


class Bounce(Game):
    def get_reward(self) -> float:
        return 0

    def get_input_shape(self) -> tuple[int, int, int, int]:
        return (-1, 6, 9, 6)
