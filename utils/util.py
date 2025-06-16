# standard libraries
from typing import List, Tuple
from pathlib import Path
from itertools import repeat

# third party libraries
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler
import numpy as np

# local packages
from utils.datatypes import Pathlike


__all__ = [
    "init_dataloader",
    "gpu_has_80GB",
    "ensure_dir",
    "inf_loop",
    "write_log",
    "prepare_device",
]


def init_dataloader(
    dataset: Dataset,
    data_loader_class: DataLoader,  # data loader class pointer
    training: bool,
    *,
    batch_size: int,
    validation_split: float = 0.0,
    batch_size_valid: int | None = None,
    **dataloader_kwargs,
) -> Tuple[DataLoader, DataLoader | None]:

    if not training or validation_split == 0:
        loader = data_loader_class(
            dataset=dataset,
            shuffle=training,
            batch_size=batch_size,
            sampler=None,
            **dataloader_kwargs,
        )
        return loader, None

    assert (
        batch_size_valid is not None
    ), "batch_size_valid must not be None when using a validatation dataloader."

    n_samples = len(dataset)
    if isinstance(validation_split, int):
        assert validation_split > 0
        assert (
            validation_split < n_samples
        ), "validation set size is configured to be larger than entire dataset."
        len_valid = validation_split
    else:
        len_valid = int(n_samples * validation_split)

    class SequentialSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    idx_full = np.arange(n_samples)
    np.random.shuffle(idx_full)

    valid_idx = idx_full[0:len_valid]
    train_idx = idx_full[len_valid:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    train_dataloader = data_loader_class(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        **dataloader_kwargs,
    )
    valid_dataloader = data_loader_class(
        dataset=dataset,
        batch_size=batch_size_valid,
        sampler=valid_sampler,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs,
    )

    return train_dataloader, valid_dataloader


def gpu_has_80GB() -> bool:
    current_device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(current_device)
    assert "A100" in device_properties.name, f"Detected {device_properties.name}"
    total_memory = device_properties.total_memory  # in bytes
    return total_memory >= 80000 * 1024**2


def ensure_dir(dirname: Pathlike) -> Path:
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
    return dirname


def inf_loop(data_loader: DataLoader):
    """Wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device() -> Tuple[torch.device, List[int]]:
    """Setup GPU device if available. get gpu device indices which are used for `DataParallel`."""
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine; training will be performed on CPU."
        )

    device = torch.device("cuda:0" if n_gpu > 0 else "cpu")
    list_ids = list(range(n_gpu))
    return device, list_ids


def prepare_device_old(n_gpu_use: int) -> Tuple[torch.device, List[int]]:
    """Setup GPU device if available. get gpu device indices which are used for `DataParallel`."""
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine; training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def write_log(*args, **kwargs) -> None:
    kwargs["mode"] = kwargs.get("mode", "a")
    with open("my_logging_file.txt", **kwargs) as f:
        text = " ".join([str(elem) for elem in args])
        f.write(text + "\n")
