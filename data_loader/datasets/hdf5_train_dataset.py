# standard libraries
from typing import List, Tuple, Literal

# third party libraries
from torch import Tensor

# local packages
from .hdf5_dataset import HDF5Dataset
from utils.transforms import get_transforms_train
from utils.datatypes import Pathlike


__all__ = [
    "HDF5TrainDataset",
]


class HDF5TrainDataset(HDF5Dataset):
    def __init__(
        self,
        filename: Pathlike,
        dataset_name_inputs: str,
        dataset_name_labels: str | None,
        dataset_name_conditions: str | None,
        hdf5_group_names: List[str],
        *,
        input_size: int | None,
        resize_method: Literal["crop", "resize"] | None,
        num_conditions: int | None = None,
        dataset_size: int | None = None,
        input_size_valid: int | None = None,
        resize_method_valid: Literal["crop", "resize"] | None = None,
    ):
        """
        Args:
            filename (Pathlike): Path to the HDF5-file containing the data.
            dataset_name_inputs (str): Name for the inputs in the dataset.
            dataset_name_labels (str | None): Name for the labels in the dataset. If None, no labels are needed.
            dataset_name_conditions (str | None): Name for the conditions (extra input channels) in the dataset.
                        If None, no further channels are added to the input.
            hdf5_group_names (List[str], optional): List of subgroups where the datasets are store. Defaults to [].
            input_size (int, optional): Input size of the training examples. If None, the original image size will be
                        used.
            resize_method (str, optional): How to resize the original image. If "crop" a random part of the image will
                        be cropped. If "resize", the image simply will be resized to the new size. If None,
                        original image will be used.
            dataset_size (int, optional): Set a custom size for the dataset. Useful, if the actual dataset is very
            input_size_valid (int, optional):  See `input_size`, only this is optional and would be applied to the
                        validation data. Defaults to to None.
            resize_method_valid (str, optional): See `resize_method`, only this is optional and would be applied to the
                        validation data. Defaults to None.
        """
        super().__init__(
            filename=filename,
            dataset_name_inputs=dataset_name_inputs,
            dataset_name_labels=dataset_name_labels,
            dataset_name_conditions=dataset_name_conditions,
            hdf5_group_names=hdf5_group_names,
            dataset_size=dataset_size,
        )
        if num_conditions is None and dataset_name_conditions is not None:
            num_conditions = 1  # default value
        self.num_conditions = num_conditions
        self.train_transform = get_transforms_train(
            input_size=input_size, resize_method=resize_method
        )
        self.valid_transform = get_transforms_train(
            input_size=input_size_valid, resize_method=resize_method_valid
        )

        self._train = True  # set to False, when doing validation

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        input, label = self.inputs[idx], self.labels[idx]
        if self.conditions is not None:
            condition = self.conditions[idx]
            if self.num_conditions is not None:
                condition = condition[: self.num_conditions]
            input = self._concat(input, condition)

        if self._train and self.train_transform:
            input, label = self.train_transform(input, label)
        elif not self._train and self.valid_transform:
            input, label = self.valid_transform(input, label)
        return input, label
