from abc import ABC, abstractmethod

from numpy.typing import NDArray
from torch import nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def predict(self, board: NDArray) -> tuple[NDArray, float]:
        """
        Base prediction method that should be implemented by subclasses
        """
        raise NotImplementedError
