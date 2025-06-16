# standard libraries

# third party libraries
import torch
import numpy as np

# local packages


__all__ = [
    "get_torch_device",
    "set_reproducibility",
]


def get_torch_device(device: str = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def set_reproducibility(seed: int | float = 69) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# def prepare_device(n_gpu_use: int):
#     """Setup GPU device if available. get gpu device indices which are used for DataParallel
#     """
#     n_gpu = torch.cuda.device_count()
#     if n_gpu_use > 0 and n_gpu == 0:
#         print("Warning: There's no GPU available on this machine," "training will be performed on CPU.")
#         n_gpu_use = 0
#     if n_gpu_use > n_gpu:
#         print(
#             f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
#             "available on this machine."
#         )
#         n_gpu_use = n_gpu
#     device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
#     list_ids = list(range(n_gpu_use))
#     return device, list_ids
