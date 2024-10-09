import torch  # type: ignore[import]
from simulator.game.connect import Config, State


def game_to_nn(config: Config, state: State) -> list[torch.Tensor]:
    tensor = torch.tensor(state.grid)  # type: ignore[arg-type]
    input_shape = (3, state.config.height, state.config.width)
    output = torch.zeros(input_shape)

    num_players = config.num_players

    for player in range(num_players):
        output[player] = (tensor == player).float()

    output[num_players] = (tensor == -1).float()
    output[num_players + 1] = (tensor == 0).float()
    output[num_players + 2] = (tensor == 1).float()

    return [output]


def get_input_shape(config: Config) -> tuple[int, int, int]:
    return (3, config.height, config.width)


def get_state_to_input(state: State) -> list[torch.Tensor]:
    tensor = torch.tensor(state.grid)  # type: ignore[arg-type]
    return [tensor]
