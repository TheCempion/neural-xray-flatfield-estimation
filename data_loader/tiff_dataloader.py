# standard libraries

# third party libraries

# local packages
from utils.datatypes import Pathlike
from utils.transforms import get_transform
from base import BaseDataLoader

from .datasets import TIFFDataset


__all__ = [
    "TIFFDataLoader",
]


class TIFFDataLoader(BaseDataLoader):
    def __init__(
        self,
        inputs: Pathlike,
        labels: Pathlike | None,
        batch_size: int,
        training: bool,
        *,
        shuffle: bool | None = None,  # if None, use training flag as shuffle
        input_size: int | None = None,  # None -> testing
        resize_method: str | None = None,  # None -> testing
        validation_split: float = 0.0,  # 0 -> default for testing
        num_workers: int = 1,
    ) -> None:
        if shuffle is None:
            shuffle = training
        transform = get_transform(
            training=training, resize_method=resize_method, input_size=input_size
        )
        self.dataset = TIFFDataset(
            inputs, training, data_dir_labels=labels, transform=transform
        )
        super().__init__(
            self.dataset,
            batch_size,
            training=training,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers,
        )
