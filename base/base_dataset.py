# standard libraries

# third party libraries
from torch.utils.data import Dataset
from torch import Tensor

# local packages


__all__ = [
    "BaseDataset",
]


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(self):
        super().__init__()
        self._train = True  # set to False, when doing validation

    def __getitem__(self, index) -> Tensor:
        raise NotImplementedError(
            "Subclasses of BaseDataset should implement __getitem__."
        )

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False
