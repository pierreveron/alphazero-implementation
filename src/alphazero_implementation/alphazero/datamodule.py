from collections import deque

import lightning as L
import torch
from torch.utils.data import DataLoader

from alphazero_implementation.alphazero.types import Episode, Sample
from alphazero_implementation.mcts.agent import MCTSAgent
from alphazero_implementation.models.model import Model


class AlphaZeroDataModule(L.LightningDataModule):
    def __init__(
        self,
        model: Model,
        agent: MCTSAgent,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        persistent_workers: bool = True,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.model = model
        self.agent = agent
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.buffer_size = buffer_size
        self.buffer: deque[Sample] = deque(maxlen=buffer_size)

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[torch.Tensor, ...]]:
        # Generate new game data
        episodes: list[Episode] = self.agent.run()

        # Add new data to the buffer
        for episode in episodes:
            self.buffer.extend(episode.samples)

        states = [sample.state for sample in self.buffer]
        policies = [sample.policy for sample in self.buffer]
        values = [sample.value for sample in self.buffer]

        dataset = self.model.format_dataset(states, policies, values)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def get_buffer_size(self) -> int:
        return len(self.buffer)
