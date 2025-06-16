# standard libraries
from typing import List, Tuple

from torch import Tensor

# third party libraries

# local packages
from .hdf5_dataset import HDF5Dataset
from utils.transforms import get_transforms_test
from utils.datatypes import Pathlike


__all__ = [
    "HDF5EvalDataset",
]


class HDF5EvalDataset(HDF5Dataset):
    """Dataset for an HDF5 file. Difference to `HDF5EvalDataset` is that this dataset also returns the gt_hologram."""

    def __init__(
        self,
        filename: Pathlike,
        dataset_name_inputs: str,
        dataset_name_labels: str | None,
        dataset_name_conditions: str | None,
        dataset_name_gt_holograms: str | None,
        hdf5_group_names: List[str] = [],
        dataset_size: int | None = None,
        num_conditions: int | None = None,
    ):
        """
        Args:
            filename (Pathlike): Path to the HDF5-file containing the data.
            dataset_name_inputs (str): Name for the inputs in the dataset.
            dataset_name_labels (str | None): Name for the labels in the dataset. If None, no labels are needed.
            dataset_name_conditions (str | None): Name for the conditions (extra input channels) in the dataset.
                        If None, no further channels are added to the input.
            dataset_name_gt_holograms (str | None): Name for the ground truth holograms, i.e. flat-field corrected
                        holograms, in the dataset. If None, GT FFC will not be used during evaluation.
            hdf5_group_names (List[str], optional): List of subgroups where the datasets are store. Defaults to [].
            dataset_size (int, optional): Set a custom size for the dataset. Useful, if the actual dataset is very
                        large. If None, the actual dataset size is used. Defaults to None.
        """
        super().__init__(
            filename=filename,
            dataset_name_inputs=dataset_name_inputs,
            dataset_name_labels=dataset_name_labels,
            dataset_name_conditions=dataset_name_conditions,
            hdf5_group_names=hdf5_group_names,
            dataset_size=dataset_size,
        )

        if dataset_name_gt_holograms is not None:
            self.gt_holograms = self.dataset[dataset_name_gt_holograms]
        else:
            self.gt_holograms = None

        if num_conditions is None and dataset_name_conditions is not None:
            num_conditions = 1  # default value
        self.num_conditions = num_conditions

        self.transform = get_transforms_test()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor | None]:
        input, label = self.inputs[idx], self.labels[idx]
        if self.conditions is not None:
            condition = self.conditions[idx]
            if self.num_conditions is not None:
                condition = condition[: self.num_conditions]
            input = self._concat(input, condition)

        input = self.transform(input)
        label = self.transform(label)

        if self.gt_holograms is not None:
            gt_hologram = self.gt_holograms[idx]
            gt_hologram = self.transform(gt_hologram)
        else:
            gt_hologram = None
        return input, label, gt_hologram
