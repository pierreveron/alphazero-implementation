from concurrent.futures import ThreadPoolExecutor

from simulator.game.connect import Config  # type: ignore[import]

from alphazero_implementation.models.games.connect4 import CNNModel
from alphazero_implementation.textual.agent import AlphaZeroAgent
from alphazero_implementation.textual.arena import ArenaApp


def play():
    config = Config(6, 7, 4)

    path = "/Users/pveron/Code/alphazero-implementation/checkpoints/run_156/model-epoch=1999.ckpt"
    path = "/Users/pveron/Code/alphazero-implementation/lightning_logs/alphazero/run_163_iter200_episodes100_sims100/checkpoints/epoch=249-step=91900.ckpt"
    path = "/Users/pveron/Code/alphazero-implementation/lightning_logs/alphazero/run_164_iter200_episodes100_sims100/checkpoints/epoch=249-step=62820.ckpt"
    path = "/Users/pveron/Code/alphazero-implementation/lightning_logs/alphazero/run_168_iter200_episodes100_sims100/checkpoints/epoch=1999-step=1431360.ckpt"
    # path = "/Users/pveron/Code/alphazero-implementation/lightning_logs/alphazero/run_095_iter50_episodes20_sims200/checkpoints/epoch=49-step=1557.ckpt"

    model = CNNModel.load_from_checkpoint(  # type: ignore[arg-type]
        path,
        height=config.height,
        width=config.width,
        max_actions=config.width,
        num_players=config.num_players,
    ).eval()

    with ThreadPoolExecutor() as executor:
        agent1 = AlphaZeroAgent(model, temperature=0)
        agent2 = AlphaZeroAgent(model)
        app = ArenaApp(agent1, agent2, executor, 2)
        app.run()


if __name__ == "__main__":
    play()
