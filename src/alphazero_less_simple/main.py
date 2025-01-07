import argparse
import cProfile
import pstats
from pathlib import Path

from alphazero_less_simple.core.training import Trainer
from alphazero_simple.config import AlphaZeroConfig
from alphazero_simple.connect4_game import Connect4Game
from alphazero_simple.resnet import ResNet


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    checkpoints = list(checkpoints_dir.glob("*.ckpt"))
    if not checkpoints:
        return None

    # Sort by modification time to get the latest
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def main(config: AlphaZeroConfig, run_dir: Path | None = None):
    game = Connect4Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = ResNet(board_size, action_size, 9, 128)

    trainer = Trainer(game, model, config)

    if run_dir is not None:
        checkpoint_path = find_latest_checkpoint(run_dir)
        samples_dir = run_dir / "samples"
        if not samples_dir.exists():
            samples_dir = None
    else:
        checkpoint_path = None
        samples_dir = None

    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Samples dir: {samples_dir}")

    trainer.learn(checkpoint_path=checkpoint_path, samples_dir=samples_dir)


def profile_train(config: AlphaZeroConfig):
    print("Profiling activated")
    profiler = cProfile.Profile()
    profiler.enable()

    main(config)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)  # Print top 20 time-consuming functions
    stats.dump_stats("train_profile.prof")  # Save profile results to a file


if __name__ == "__main__":
    config = AlphaZeroConfig(
        batch_size=64,
        num_iterations=20,
        num_simulations=600,
        num_episodes=5000,
        num_parallel_episodes=100,
        num_iters_for_train_history=1,
        epochs=1,
        mem_buffer_size=int(5e5),
        background_generation=False,
    )

    parser = argparse.ArgumentParser(description="Train the AlphaZero model")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a previous run directory to resume from (e.g. lightning_logs/alphazero_less_simple/run_XXX...)",
    )
    args = parser.parse_args()

    if args.profile:
        profile_train(config)
    else:
        main(config, run_dir=args.run_dir)
