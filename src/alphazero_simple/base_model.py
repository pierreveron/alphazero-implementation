from abc import ABC, abstractmethod

from numpy.typing import NDArray
from torch import nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def predict(self, boards: list[NDArray]) -> tuple[list[NDArray], list[float]]:
        """
        Base prediction method that should be implemented by subclasses
        """
        raise NotImplementedError
