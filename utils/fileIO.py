# standard libraries
import json
from pathlib import Path
from typing import Dict, Any, Optional
from collections import OrderedDict

# third party libraries
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# local packages
import utils.constants as const
from .datatypes import Pathlike
from .util import ensure_dir


__all__ = [
    "load_img",
    "save_img",
    "read_img",
    "write_img",
    "savefig",
]


def load_img(
    path: Pathlike, expand_dim: bool = True, convert_to_float32: bool = True
) -> np.ndarray:
    img = io.imread(path)
    if expand_dim and img.ndim == 2:
        img = np.expand_dims(img, axis=0)
    if convert_to_float32:
        img = img.astype(np.float32)
    return img


def save_img(path: Pathlike, img: np.ndarray, plugin: Optional[bool] = None) -> None:
    if img.dtype == np.float32 or img.dtype == np.float64:
        plugin = "tifffile"
    io.imsave(path, img, plugin=plugin)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Dict[Any, Any], fname: Pathlike):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def savefig(
    output_dir: Path, idx: int, close_fig: bool = True, fig: plt.Figure | None = None
) -> None:
    """Wrapper to save a plt-figure."""
    ensure_dir(output_dir)
    if fig is not None:
        fig.savefig(output_dir / f"{idx:03}.{const.FILE_EXT}")
    else:
        plt.savefig(output_dir / f"{idx:03}.{const.FILE_EXT}")
    if close_fig:
        plt.close(fig)
