# standard libraries

# third party libraries
from torch.utils.data import DataLoader

from torch.utils.data.dataloader import _collate_fn_t
from torch.utils.data.sampler import Sampler

# local packages
from .base_dataset import BaseDataset


__all__ = [
    "BaseDataLoader",
]


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler | None,
        drop_last: bool = False,
        n_samples: int | None = None,
        collate_fn: _collate_fn_t = None,
    ):
        assert not (shuffle and bool(sampler))

        self.shuffle = shuffle
        self.batch_idx = 0
        if n_samples is None:
            self.n_samples = len(dataset)
        else:
            self.n_samples = n_samples

        self.sampler = sampler

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": batch_size,
            "drop_last": drop_last,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def train(self) -> None:
        self.dataset.train()

    def eval(self) -> None:
        self.dataset.eval()
