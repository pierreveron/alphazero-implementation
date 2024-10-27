import cProfile
import pstats

from simulator.game.connect import Config  # type: ignore[import]

from alphazero_implementation.alphazero.trainer import AlphaZeroTrainer
from alphazero_implementation.models.games.connect4.v1 import BasicNN


def train():
    config = Config(6, 7, 4)
    initial_state = config.sample_initial_state()

    model = BasicNN(
        height=config.height,
        width=config.width,
        max_actions=config.width,
        num_players=config.num_players,
    )

    trainer = AlphaZeroTrainer(
        model=model,
        simulations_per_episode=200,
        episodes_per_iter=20,
    )

    trainer.train(
        num_iterations=50,
        epochs_per_iter=10,
        initial_state=initial_state,
    )


def profile_train():
    profiler = cProfile.Profile()
    profiler.enable()

    train()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)  # Print top 20 time-consuming functions
    stats.dump_stats("train_profile.prof")  # Save profile results to a file


if __name__ == "__main__":
    profile_train()
