# standard libraries
from typing import Any, Tuple
from functools import wraps

# third party libraries
from matplotlib import rc, rcParams
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# local packages
import utils.constants as const
import utils.plotting.mpl_constants as mpl_constants
from utils.datatypes import Pathlike


__all__ = [
    "draw_rectangle",
    "set_border",
    "get_figsize",
    "subplots",
    "axhline_wrapper",
    "init_matplotlib",
    "change_mpl_settings",
]


def draw_rectangle(
    ax: Any,
    /,
    anchor: Tuple[int, int],
    height: int,
    width: int,
    *,
    color: str = "r",
    lw: int = 1,
    **kwargs,
) -> None:
    """Draws a rectangle in an image.

    Wrapper for matplotlib.patches.Rectangle implementation. The MPL doc says "The rectangle extends from xy[0] to x[0]
    + width in x-direction and from xy[1] to xy[1] + height in y-direction." For convenience reasons, the anchor point will be treated as in image coordinates, i.e. anchor[0] corresponds to the y-axis and and anchor[1] to the x-axis.

    Args:
        ax (Any): Axis on which the rectangle will be drawn.
        anchor (Tuple[int, int]): Upper left corner of rectangle (w.r.t. image coordinate system).
        height (int): Height of the rectangle.
        width (int): Width of the rectangle.
        color (str, optional): Color of the rectangle. Defaults to "r".
        lw (int, optional): Linewidth of the rectangle. Defaults to 1.
    """
    kwargs["facecolor"] = kwargs.get("facecolor", "none")
    anchor = (anchor[1], anchor[0])
    rect = patches.Rectangle(
        xy=anchor, width=width, height=height, linewidth=lw, edgecolor=color, **kwargs
    )
    ax.add_patch(rect)  # Add the patch to the Axes
    rect.set_clip_on(False)


def set_border(ax: Any, *, lw: int = 2, color: str = "red") -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)  # Adjust thickness as needed


def get_figsize(
    rows: int = 1, cols: int | None = None, figsize: int = const.FIG_SIZE
) -> Tuple[int, int]:
    if cols is None:
        cols = rows
    return (cols * figsize, rows * figsize)


def subplots(
    rows: int = 1, cols: int = 1, *, constrained_layout: bool = True, **kwargs
) -> Tuple[plt.Figure, Any]:
    return plt.subplots(
        rows,
        cols,
        figsize=get_figsize(rows, cols),
        constrained_layout=constrained_layout,
        **kwargs,
    )


def axhline_wrapper(line: int, color: str | None = None, **kwargs) -> callable:
    if (
        color is not None
    ):  # circumvent to pass `mpl_constants.LATEX_COLORS` to this function.
        kwargs["color"] = mpl_constants.LATEX_COLORS.get("color", color)
    return lambda ax: ax.axhline(line, **kwargs)


def init_matplotlib():
    rc("font", family="serif")
    rc("text", usetex=True)
    rc("image", cmap=const.CMAP)
    rc("savefig", format=const.FILE_EXT)
    rc("latex")
    rcParams.update(mpl_constants.MPL_FONT_SIZE_DEFAULT)
    rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    # TODO: Set the color cycle to LaTeX-like colors
    # plt.rc("axes", prop_cycle=cycler("color", list(mpl_constants.LATEX_COLORS.values())))


def change_mpl_settings(
    group: str, /, *, restore_previous: bool = True, **kwargs
) -> callable:
    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args, **func_kwargs) -> Any:
            # Store the current settings to revert after function call
            previous_settings = {
                f"{group}.{key}": rcParams[f"{group}.{key}"] for key in kwargs
            }
            rc(group, **kwargs)
            result = func(*args, **func_kwargs)
            if restore_previous:
                rcParams.update(previous_settings)

            return result

        return wrapper

    return decorator
