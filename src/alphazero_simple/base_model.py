from abc import ABC, abstractmethod

import torch
from numpy.typing import NDArray
from torch import nn


class BaseModel(nn.Module, ABC):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    @abstractmethod
    def predict(self, board: NDArray) -> tuple[NDArray, float]:
        """
        Base prediction method that should be implemented by subclasses
        """
        raise NotImplementedError
