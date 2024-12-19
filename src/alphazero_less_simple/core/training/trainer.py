import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from alphazero_simple.base_game import BaseGame
from alphazero_simple.base_model import BaseModel
from alphazero_simple.config import AlphaZeroConfig
from alphazero_simple.lightning_model import AlphaZeroLitModule

from . import DataModule, EpisodeGenerator


class Trainer:
    def __init__(
        self,
        game: BaseGame,
        model: BaseModel,
        config: AlphaZeroConfig,
    ):
        self.game = game
        self.model = model
        self.config = config

    def _get_next_run_number(self):
        base_dir = "lightning_logs/alphazero_less_simple"
        if not os.path.exists(base_dir):
            return 1
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
        if not existing_runs:
            return 1
        return max(int(run.split("_")[1]) for run in existing_runs) + 1

    def learn(
        self,
    ):
        # Create a consistent run name
        run_counter = self._get_next_run_number()
        run_name = f"run_{run_counter:03d}_{self.model.__class__.__name__}_iter{self.config.num_iterations}_episodes{self.config.num_episodes}_sims{self.config.num_simulations}"

        # Create logger with the run name
        logger = TensorBoardLogger(
            "lightning_logs",
            name="alphazero_less_simple",
            version=run_name,
        )

        episode_generator = EpisodeGenerator(
            game=self.game,
            config=self.config,
        )

        # Create data module with the same run name
        datamodule = DataModule(
            model=self.model,
            episode_generator=episode_generator,
            config=self.config,
            save_dir=f"lightning_logs/alphazero_less_simple/{run_name}/episodes",
        )

        # Create checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            # filename="{epoch}-{train_loss:.2f}",
            every_n_epochs=self.config.epochs
            * int(self.config.num_iters_for_train_history / 2),
            save_top_k=-1,  # Keep all checkpoints
        )

        # Create trainer with checkpoint callback
        trainer = L.Trainer(
            max_epochs=self.config.num_iterations * self.config.epochs,
            log_every_n_steps=1,
            enable_progress_bar=True,
            logger=logger,
            reload_dataloaders_every_n_epochs=self.config.epochs,
            callbacks=[checkpoint_callback],
        )

        lit_model = AlphaZeroLitModule(self.model)

        # Train the model
        trainer.fit(lit_model, datamodule=datamodule)

        print("Training completed!")
