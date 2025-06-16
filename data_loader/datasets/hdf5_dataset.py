# standard libraries
from typing import List, Tuple
from abc import abstractmethod

# third party libraries
import numpy as np
from torch import Tensor
import h5py

# local packages
from utils.datatypes import Pathlike
from base import BaseDataset


__all__ = [
    "HDF5Dataset",
]


class HDF5Dataset(BaseDataset):
    """Dataset for an HDF5 file."""

    def __init__(
        self,
        filename: Pathlike,
        dataset_name_inputs: str,
        dataset_name_labels: str | None,
        dataset_name_conditions: str | None,
        hdf5_group_names: List[str] = [],
        dataset_size: int | None = None,
    ):
        """
        Args:
            filename (Pathlike): Path to the HDF5-file containing the data.
            dataset_name_inputs (str): Name for the inputs in the dataset.
            dataset_name_labels (str | None): Name for the labels in the dataset. If None, no labels are needed.
            dataset_name_conditions (str | None): Name for the conditions (extra input channels) in the dataset.
                        If None, no further channels are added to the input.
            hdf5_group_names (List[str], optional): List of subgroups where the datasets are store. Defaults to [].
            dataset_size (int, optional): Set a custom size for the dataset. Useful, if the actual dataset is very
                        large. If None, the actual dataset size is used. Defaults to None.
        """
        super().__init__()
        self.hdf5 = h5py.File(filename, "r")
        if hdf5_group_names != []:
            sub_group = self.hdf5
            for group_name in hdf5_group_names:
                sub_group = sub_group[group_name]
            self.dataset = sub_group
        else:
            self.dataset = self.hdf5

        self.inputs = self.dataset[dataset_name_inputs]
        if dataset_name_labels is not None:
            self.labels = self.dataset[dataset_name_labels]
        else:
            self.labels = None

        if dataset_name_conditions is not None:
            self.conditions = self.dataset[dataset_name_conditions]
        else:
            self.conditions = None

        self.length = len(self.inputs)
        if dataset_size is not None:
            assert dataset_size <= len(self.inputs)
            self.length = dataset_size

    def __del__(self):
        try:
            self.hdf5.close()
        except AttributeError:
            print("Cannot close hdf5 file in destructor.")

    def __len__(self) -> int:
        return self.length

    def close(self) -> None:
        self.hdf5.close()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        raise NotImplementedError("Method needs to be implenented in child classes.")

    def _concat(self, a1: np.ndarray, a2: np.ndarray, new_dim: int = 3) -> np.ndarray:
        while a1.ndim < new_dim:
            a1 = np.expand_dims(a1, axis=0)
        while a2.ndim < new_dim:
            a2 = np.expand_dims(a2, axis=0)
        return np.concatenate([a1, a2], axis=0)
