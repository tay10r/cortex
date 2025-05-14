from abc import ABC, abstractmethod
from torch import Tensor, nn


class Observer(ABC):
    """
    This class observes the training loop.
    It can be used for various reporting and logging tasks.
    All base class methods are optional to override.
    """

    @abstractmethod
    def on_model_update(self, net: nn.Module, epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def on_scalar_metric(self, name: str, value: float, epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def on_preview(self, grid: Tensor, epoch: int):
        raise NotImplementedError()

    @abstractmethod
    def on_complete(self):
        raise NotImplementedError()
