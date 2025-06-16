# standard libraries

# third party libraries
from torch.utils.data import Sampler

# local packages
from base import BaseDataLoader
from .datasets import HDF5Dataset


__all__ = [
    "HDF5DataLoader",
]


class HDF5DataLoader(BaseDataLoader):
    """Custom DataLoader for holograms and corresponding flatfields."""

    def __init__(
        self,
        *,
        dataset: HDF5Dataset,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler | None,
        drop_last: bool = False,
        n_samples: int | None = None,
        **kwargs
    ) -> None:

        self.dataset = (
            dataset  # also store here for linting and typehinting of correct Dataset
        )
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=drop_last,
            n_samples=n_samples,
            **kwargs,
        )
