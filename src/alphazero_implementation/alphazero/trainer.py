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
        mcgs_num_simulations: int = 800,
        games_per_iteration: int = 100,
    ):
        self.model = model
        self.mcgs_num_simulations = mcgs_num_simulations
        self.games_per_iteration = games_per_iteration
        self.run_counter = self.get_next_run_number()
        self.mcts_agent = MCTSAgent(
            model=self.model,
            self_play_count=self.games_per_iteration,
            num_simulations_per_self_play=self.mcgs_num_simulations,
        )

    def get_next_run_number(self):
        base_dir = "lightning_logs/alphazero"
        if not os.path.exists(base_dir):
            return 1
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
        if not existing_runs:
            return 1
        return max(int(run.split("_")[1]) for run in existing_runs) + 1

    def train(
        self,
        iterations: int,
        initial_state: State,
        num_playouts: int,
    ):
        # Create logger
        logger = TensorBoardLogger(
            "lightning_logs",
            name="alphazero",
            version=f"run_{self.run_counter:03d}_iter{iterations}_sims{self.mcgs_num_simulations}_batch{self.games_per_iteration}",
        )

        # Create data module
        datamodule = AlphaZeroDataModule(
            model=self.model,
            agent=self.mcts_agent,
            initial_state=initial_state,
            num_playouts=num_playouts,
            games_per_iteration=self.games_per_iteration,
        )

        # Create trainer
        trainer = L.Trainer(
            max_epochs=iterations,  # Now each epoch represents one iteration
            log_every_n_steps=1,
            enable_progress_bar=True,
            logger=logger,
            reload_dataloaders_every_n_epochs=1,
        )

        # Train the model
        trainer.fit(self.model, datamodule=datamodule)

        print("Training completed!")
