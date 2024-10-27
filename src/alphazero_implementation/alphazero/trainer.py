import os

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from simulator.game.connect import State  # type: ignore[import]

from alphazero_implementation.alphazero.datamodule import AlphaZeroDataModule
from alphazero_implementation.mcts.agent import MCTSAgent
from alphazero_implementation.models.model import Model


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
    ):
        # Create logger
        logger = TensorBoardLogger(
            "lightning_logs",
            name="alphazero",
            version=f"run_{self.run_counter:03d}_iter{num_iterations}_episodes{self.episodes_per_iter}_sims{self.simulations_per_episode}",
        )

        mcts_agent = MCTSAgent(
            model=self.model,
            num_episodes=self.episodes_per_iter,
            simulations_per_episode=self.simulations_per_episode,
            initial_state=initial_state,
        )

        # Create data module
        datamodule = AlphaZeroDataModule(
            model=self.model,
            agent=mcts_agent,
        )

        # Create trainer
        trainer = L.Trainer(
            max_epochs=num_iterations * epochs_per_iter,
            log_every_n_steps=1,
            enable_progress_bar=True,
            logger=logger,
            reload_dataloaders_every_n_epochs=epochs_per_iter,
        )

        # Train the model
        trainer.fit(self.model, datamodule=datamodule)

        print("Training completed!")
