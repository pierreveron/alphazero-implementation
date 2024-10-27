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
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.model = model
        self.agent = agent
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        batch_data: GameHistory = self.agent.run()

        states: list[State]
        policies: list[ActionPolicy]
        values: list[list[float]]
        states, policies, values = zip(*batch_data)

        dataset = self.model.format_dataset(states, policies, values)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
