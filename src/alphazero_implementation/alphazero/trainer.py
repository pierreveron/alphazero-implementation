import os

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from simulator.game.connect import State  # type: ignore[import]
from torch.utils.data import DataLoader

from alphazero_implementation.mcts.agent import MCTSAgent
from alphazero_implementation.models.model import ActionPolicy, Model

# GameHistory represents the trajectory of a single game
# It is a list of tuples, where each tuple contains:
# - State: The game state at that point
# - list[float]: The improved policy (action probabilities) for that state
# - list[float]: The value (expected outcome) for each player at that state
GameHistory = list[tuple[State, ActionPolicy, list[float]]]


class AlphaZeroTrainer:
    def __init__(
        self,
        model: Model,
        mcgs_num_simulations: int = 800,
        mcgs_self_play_count: int = 100,
    ):
        self.model = model
        self.mcgs_num_simulations = mcgs_num_simulations
        self.mcgs_self_play_count = mcgs_self_play_count
        self.run_counter = self.get_next_run_number()
        self.mcts_agent = MCTSAgent(
            model=self.model,
            self_play_count=self.mcgs_self_play_count,
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
        num_iterations: int,
        initial_state: State,
        max_epochs: int = 100,
    ):
        training_data: GameHistory = []

        # Create a TensorBoard logger with version name including parameters and run number as prefix
        logger = TensorBoardLogger(
            "lightning_logs",
            name="alphazero",
            version=f"run_{self.run_counter:03d}_iter{num_iterations}_sims{self.mcgs_num_simulations}_batch{self.mcgs_self_play_count}",
        )

        trainer = L.Trainer(
            max_epochs=max_epochs,
            log_every_n_steps=10,
            enable_progress_bar=True,
            logger=logger,  # Add the logger to the Trainer
        )

        # tuner = Tuner(trainer)

        # Deepmind's AlphaZero pseudocode continuously train the model as an optimization
        # process, but we choose to do this in smaller batches
        for iteration in range(num_iterations):
            trajectories = self.mcts_agent.run_self_plays(initial_state)
            training_data.extend([item for sublist in trajectories for item in sublist])
            # TODO: remove old training data

            states, policies, values = zip(*training_data)

            dataset = self.model.format_dataset(states, policies, values)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # tuner.lr_find(self.model)

            trainer.fit(self.model, dataloader)

            print(f"Iteration [{iteration+1}/{num_iterations}] completed!")

        print("Training completed!")
