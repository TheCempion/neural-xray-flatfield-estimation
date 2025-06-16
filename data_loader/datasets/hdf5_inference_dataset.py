# standard libraries
from typing import List

# third party libraries
from torch import Tensor
import numpy as np

# local packages
from .hdf5_dataset import HDF5Dataset
from utils.transforms import get_transforms_test
from utils.datatypes import Pathlike


__all__ = [
    "HDF5InferenceDataset",
]


class HDF5InferenceDataset(HDF5Dataset):
    """Dataset for an HDF5 file for real world data without a target label."""

    def __init__(
        self,
        filename: Pathlike,
        dataset_name_inputs: str,
        dataset_name_conditions: str | None,
        hdf5_group_names: List[str] = [],
        dataset_size: int | None = None,
        use_subset: bool = True,
        fixed_condition_idx: (
            int | None
        ) = 0,  # TODO: 6 for spiderhair; 84 magnesium wire
    ):
        """
        Args:
            filename (Pathlike): Path to the HDF5-file containing the data.
            dataset_name_inputs (str): Name for the inputs in the dataset.
            dataset_name_conditions (str | None): Name for the conditions (extra input channels) in the dataset.
                        If None, no further channels are added to the input.
            hdf5_group_names (List[str], optional): List of subgroups where the datasets are store. Defaults to [].
            dataset_size (int, optional): Set a custom size for the dataset. Useful, if the actual dataset is very
                        large. If None, the actual dataset size is used. Defaults to None.
            use_subset (bool, optional): If True use a pre-selected subset of holograms. Defaults to True.
            fixed_condition_idx (int, optional): If index is not None, use that as the condition for all holograms.
                        Otherwise use a random index. Defaults to 0.
        """
        if use_subset and not dataset_name_inputs.endswith("_subset"):
            dataset_name_inputs += "_subset"

        super().__init__(
            filename=filename,
            dataset_name_inputs=dataset_name_inputs,
            dataset_name_labels=None,
            dataset_name_conditions=dataset_name_conditions,
            hdf5_group_names=hdf5_group_names,
            dataset_size=dataset_size,
        )
        self.transform = get_transforms_test()
        assert (
            dataset_name_conditions is None
            or fixed_condition_idx is None
            or fixed_condition_idx < len(self.conditions)
        ), f"{ len(self.conditions)=}, {fixed_condition_idx}"
        self.fixed_condition_idx = fixed_condition_idx

    def __getitem__(self, idx: int) -> Tensor:
        input = self.inputs[idx]
        if self.conditions is not None:
            if self.fixed_condition_idx is not None:
                condition = self.conditions[self.fixed_condition_idx]
            else:
                cond_idx = np.random.choice(len(self.conditions))
                condition = self.conditions[cond_idx]
            input = self._concat(input, condition)
        return self.transform(input)
