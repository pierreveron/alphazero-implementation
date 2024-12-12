import argparse
import cProfile
import pstats

from simulator.game.connect import Config  # type: ignore[import]

from alphazero_implementation.core.training import Trainer
from alphazero_implementation.models.games.connect4 import CNNModel


def train():
    config = Config(6, 7, 4)
    initial_state = config.sample_initial_state()

    # Define hyperparameters
    num_iterations = 200
    epochs_per_iter = 10
    simulations_per_episode = 100
    episodes_per_iter = 100
    save_every_n_iterations = 10
    buffer_size = episodes_per_iter * save_every_n_iterations

    model = CNNModel(
        height=config.height,
        width=config.width,
    )

    # Save hyperparameters
    model.save_hyperparameters(
        {
            "training": {
                "num_iterations": num_iterations,
                "epochs_per_iter": epochs_per_iter,
                "episodes_per_iter": episodes_per_iter,
                "simulations_per_episode": simulations_per_episode,
                "buffer_size": buffer_size,
                "save_every_n_iterations": save_every_n_iterations,
            }
        }
    )

    trainer = Trainer(
        model=model,
    )

    trainer.train(
        num_iterations=num_iterations,
        epochs_per_iter=epochs_per_iter,
        episodes_per_iter=episodes_per_iter,
        simulations_per_episode=simulations_per_episode,
        initial_state=initial_state,
        buffer_size=buffer_size,
        save_every_n_iterations=save_every_n_iterations,
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
    parser = argparse.ArgumentParser(description="Train the AlphaZero model")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    if args.profile:
        profile_train()
    else:
        train()
