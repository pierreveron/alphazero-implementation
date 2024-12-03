import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from simulator.game.connect import State  # type: ignore[import]

from alphazero_implementation.alphazero.datamodule import AlphaZeroDataModule
from alphazero_implementation.models.model import Model
from alphazero_implementation.search.mcts import AlphaZeroMCTS


class AlphaZeroTrainer:
    def __init__(
        self,
        model: Model,
        episodes_per_iter: int = 100,
        simulations_per_episode: int = 800,
    ):
        self.model = model
        self.simulations_per_episode = simulations_per_episode
        self.episodes_per_iter = episodes_per_iter
        self.run_counter = self._get_next_run_number()

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
        num_iterations: int,
        epochs_per_iter: int,
        initial_state: State,
        buffer_size: int,
        save_every_n_iterations: int,
    ):
        # Create logger
        logger = TensorBoardLogger(
            "lightning_logs",
            name="alphazero",
            version=f"run_{self.run_counter:03d}_iter{num_iterations}_episodes{self.episodes_per_iter}_sims{self.simulations_per_episode}",
        )

        # mcts_agent = MCTSAgent(
        #     model=self.model,
        #     num_episodes=self.episodes_per_iter,
        #     simulations_per_episode=self.simulations_per_episode,
        #     initial_state=initial_state,
        #     parallel_mode=True,
        # )
        mcts_agent = AlphaZeroMCTS(
            model=self.model,
            num_simulations=self.simulations_per_episode,
            num_episodes=self.episodes_per_iter,
            game_initial_state=initial_state,
        )

        # Create data module
        datamodule = AlphaZeroDataModule(
            model=self.model,
            agent=mcts_agent,
            buffer_size=buffer_size,
            save_every_n_iterations=save_every_n_iterations,
            save_dir=f"lightning_logs/alphazero/run_{self.run_counter:03d}_iter{num_iterations}_episodes{self.episodes_per_iter}_sims{self.simulations_per_episode}/episodes",
        )

        # Create checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            # filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
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
            callbacks=[checkpoint_callback],  # Add the checkpoint callback
        )

        # Train the model
        trainer.fit(self.model, datamodule=datamodule)

        print("Training completed!")
