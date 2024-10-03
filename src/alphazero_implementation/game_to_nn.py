import torch  # type: ignore[import]
from simulator.game.connect import Config, State


def game_to_nn(config: Config, state: State) -> list[torch.Tensor]:
    tensor = torch.tensor(state.get_tensors()[0])
    input_shape = (3, state.config.height, state.config.width)
    output = torch.zeros(input_shape)

    num_players = config.num_players

    for player in range(num_players):
        output[player] = (tensor == player).float()

    output[num_players] = (tensor == -1).float()
    output[num_players + 1] = (tensor == 0).float()
    output[num_players + 2] = (tensor == 1).float()

    return [output]
