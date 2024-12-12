from typing import Optional

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AlphaZeroDataset(Dataset):
    def __init__(self, examples: list[tuple[np.ndarray, np.ndarray, float]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board, pi, v = self.examples[idx]
        return (
            torch.FloatTensor(board.astype(np.float64)),
            torch.FloatTensor(pi),
            torch.FloatTensor([v]).squeeze(),
        )


class AlphaZeroDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_examples: list[tuple[np.ndarray, np.ndarray, float]],
        batch_size: int,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_examples = train_examples
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = AlphaZeroDataset(self.train_examples)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
