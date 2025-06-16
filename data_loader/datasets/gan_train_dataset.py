# standard libraries
from typing import Tuple

# third party libraries
from torch import Tensor

# local packages
from .hdf5_train_dataset import HDF5TrainDataset


__all__ = [
    "GANTrainDataset",
]


class GANTrainDataset(HDF5TrainDataset):
    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: Positional arguments for `HDF5TrainDataset`.
            **kwargs: Keyword arguments for `HDF5TrainDataset`.
        """
        super().__init__(*args, **kwargs)
        self._train = True

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor | None]:
        """Get input, target and condition separately (not concatenated)."""
        input, label = super().__getitem__(idx=idx)
        if len(input) > 1:
            input, condition = input[:1].clone(), input[1:].clone()
        else:
            condition = None
        return input, label, condition
