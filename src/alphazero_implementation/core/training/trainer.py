import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from simulator.game.connect import State  # type: ignore[import]

from alphazero_implementation.core.training import DataModule, EpisodeGenerator
from alphazero_implementation.models.base import Model


class Trainer:
    def __init__(
        self,
        model: Model,
    ):
        self.model = model

    def _get_next_run_number(self):
        base_dir = "lightning_logs/alphazero"
        if not os.path.exists(base_dir):
            return 1
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
        if not existing_runs:
            return 1
        return max(int(run.split("_")[1]) for run in existing_runs) + 1

    def train(
        self,
        *,
        num_iterations: int,
        episodes_per_iter: int,
        simulations_per_episode: int,
        epochs_per_iter: int,
        initial_state: State,
        buffer_size: int,
        save_every_n_iterations: int,
    ):
        # Create a consistent run name
        run_counter = self._get_next_run_number()
        run_name = f"run_{run_counter:03d}_{self.model.__class__.__name__}_iter{num_iterations}_episodes{episodes_per_iter}_sims{simulations_per_episode}"

        # Create logger with the run name
        logger = TensorBoardLogger(
            "lightning_logs",
            name="alphazero",
            version=run_name,
        )

        episode_generator = EpisodeGenerator(
            model=self.model,
            num_simulations=simulations_per_episode,
            num_episodes=episodes_per_iter,
            game_initial_state=initial_state,
        )

        # Create data module with the same run name
        datamodule = DataModule(
            model=self.model,
            episode_generator=episode_generator,
            buffer_size=buffer_size,
            save_every_n_iterations=save_every_n_iterations,
            save_dir=f"lightning_logs/alphazero/{run_name}/episodes",
        )

        # Create checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            # filename="{epoch}-{train_loss:.2f}",
            every_n_epochs=save_every_n_iterations * epochs_per_iter,
            save_top_k=-1,  # Keep all checkpoints
        )

        # Create trainer with checkpoint callback
        trainer = L.Trainer(
            max_epochs=num_iterations * epochs_per_iter,
            log_every_n_steps=1,
            enable_progress_bar=True,
            logger=logger,
            reload_dataloaders_every_n_epochs=epochs_per_iter,
            callbacks=[checkpoint_callback],
        )

        # Train the model
        trainer.fit(self.model, datamodule=datamodule)

        print("Training completed!")
