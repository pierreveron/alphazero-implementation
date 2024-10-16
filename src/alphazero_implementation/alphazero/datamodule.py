import lightning as L
import torch
from simulator.game.connect import State  # type: ignore[import]
from torch.utils.data import DataLoader

from alphazero_implementation.mcts.agent import GameHistory, MCTSAgent
from alphazero_implementation.models.model import ActionPolicy, Model


class AlphaZeroDataModule(L.LightningDataModule):
    def __init__(
        self,
        model: Model,
        agent: MCTSAgent,
        initial_state: State,
        num_playouts: int,
        games_per_iteration: int,
        batch_size: int = 32,
        num_workers: int = 7,
    ):
        super().__init__()
        self.model = model
        self.agent = agent
        self.initial_state = initial_state
        self.num_playouts = num_playouts
        self.games_per_iteration = games_per_iteration
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        batch_data: GameHistory = []

        # Generate one iteration of games
        for _ in range(self.games_per_iteration):
            game_data = self.agent.self_play(
                self.initial_state,
                self.num_playouts,
            )
            batch_data.extend(game_data)

        states: list[State]
        policies: list[ActionPolicy]
        values: list[list[float]]
        states, policies, values = zip(*batch_data)

        dataset = self.model.format_dataset(states, policies, values)
        return DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
