from concurrent.futures import ThreadPoolExecutor

from simulator.game.connect import Config  # type: ignore[import]

from alphazero_implementation.models.games.connect4 import CNNModel

# from simulator.textual.examples.agent import AgentApp, RandomAgent
from alphazero_implementation.textual.agent import AlphaZeroAgent, RandomAgent
from alphazero_implementation.textual.arena import ArenaApp


def play():
    config = Config(6, 7, 4)

    path = "/Users/pveron/Code/alphazero-implementation/lightning_logs/alphazero/run_137_iter10_episodes10_sims50/checkpoints/epoch=99-step=20390.ckpt"

    model = CNNModel.load_from_checkpoint(  # type: ignore[arg-type]
        path,
        height=config.height,
        width=config.width,
        max_actions=config.width,
        num_players=config.num_players,
    )

    model.eval()

    with ThreadPoolExecutor() as executor:
        agent1 = AlphaZeroAgent(model)
        agent2 = RandomAgent()
        app = ArenaApp(agent1, agent2, executor)
        app.run()


if __name__ == "__main__":
    play()
