# standard libraries
import glob
from typing import Tuple

# third party libraries
from torch import Tensor
from torch.utils.data import Dataset

# local packages
import utils.fileIO as fileIO
from utils.datatypes import Pathlike, Transform


__all__ = [
    "TIFFDataset",
]


class TIFFDataset(Dataset):
    """Dataset for TIFF-files."""

    def __init__(
        self,
        data_dir_inputs: Pathlike,
        /,
        training: bool,
        *,
        data_dir_labels: Pathlike | None = None,
        transform: Transform = None,
    ) -> None:
        """
        Args:
            data_dir_inputs (Pathlike): Data directory containing the inputs.
            training (bool, optional): Determines which data will be loaded.
            data_dir_labels (Pathlike, optional): Data directory containing the labels. Defaults to `None`.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to `None`.
        """
        super().__init__()
        self.data_dir_inputs = data_dir_inputs
        self.data_dir_labels = data_dir_labels
        self._get_item = self._get_item_train if training else self._get_item_test
        self.sample_files = sorted(list(glob.glob(f"{data_dir_inputs}/*.tiff")))
        self.transform = transform
        self.training = training
        if self.training:
            self.target_files = sorted(list(glob.glob(f"{data_dir_labels}/*.tiff")))

    def _get_item_train(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = fileIO.load_img(self.sample_files[idx])
        label = fileIO.load_img(self.target_files[idx])
        if self.transform:
            sample, label = self.transform(sample, label)
        return sample, label

    def _get_item_test(self, idx: int) -> Tensor:
        sample = fileIO.load_img(self.sample_files[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self._get_item(idx)

    def __iadd__(self, other):
        if not isinstance(other, TIFFDataset):
            raise ValueError(f"Cannot add type {type(other)} to type {type(self)}.")
        self.sample_files += other.sample_files
        if self.training:
            self.target_files += other.target_files
        return self
