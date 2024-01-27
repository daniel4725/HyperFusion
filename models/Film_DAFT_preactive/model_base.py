from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

from torch.nn import Module


def check_is_unique(values: Sequence[Any]) -> bool:
    if len(values) != len(set(values)):
        raise ValueError("values of list must be unique")


class BaseModel(Module, metaclass=ABCMeta):
    """Abstract base class for models that can be executed by
    :class:`daft.training.train_and_eval.ModelRunner`.
    """

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        check_is_unique(self.input_names)
        check_is_unique(self.output_names)
        check_is_unique(list(self.input_names) + list(self.output_names))

    @property
    @abstractmethod
    def input_names(self) -> Sequence[str]:
        """Names of parameters passed to self.forward"""

    @property
    @abstractmethod
    def output_names(self) -> Sequence[str]:
        """Names of tensors returned by self.forward"""