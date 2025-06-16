# standard libraries
from pathlib import Path
from typing import Callable, Tuple

# third party libraries
from torch import Tensor
import numpy as np

# local packages


__all__ = [
    "Pathlike",
    "Tensorlike",
    "Transform",
    "callback_t",
    "stats_callback_t",
    "Loss_t",
]


Pathlike = str | Path
Tensorlike = Tensor | np.ndarray
Transform = Callable[
    [Tensorlike | Tuple[Tensorlike, Tensorlike]],
    Tensorlike | Tuple[Tensorlike, Tensorlike],
]
callback_t = Callable[[int], None]
stats_callback_t = Callable[[None], None]
Loss_t = Callable[[Tensor, Tensor], Tensor]
