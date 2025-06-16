# standard libraries
from typing import Any, Tuple, List, Callable
from pathlib import Path

# third party libraries
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.cm as cm
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

# local packages
import utils.constants as const
from .utils import get_figsize, subplots
import utils.plotting.mpl_constants as mpl_constants
from utils.datatypes import Tensorlike, Pathlike
import utils.flatfield_correction as ffc
from utils import ensure_dir


__all__ = [
    "subplot_and_store",
    "histogram",
    "colorbar",
    "imshow_colorbar",
    "imsave_min_max",
    "plot_line_values",
    "plot_single_img",
    "show_difference_map",
    "plot_grid",
    "plot_ffc_process",
]


def subplot_and_store(
    ax: axes.Axes,
    /,
    plot_func: Callable[[axes.Axes, plt.Figure, Any], Any],
    title: str,
    *,
    output_dir: Path,
    filename: int,
    xlabel: str | None = None,
    ylabel: str | None = None,
    axhline_wrapper: (
        Callable[[axes.Axes, Any], axes.Axes] | None
    ) = None,  # actually plotting.utils -> axhline_wrapper
    **kwargs,
) -> None:
    # first store the single image
    fig_single, ax_single = subplots(1, 1)
    plot_func(ax_single, **kwargs)
    if axhline_wrapper is not None:
        axhline_wrapper(ax_single)
    ax_single.set_xlabel(xlabel)
    ax_single.set_ylabel(ylabel)
    output_single_plot_file = (
        ensure_dir(output_dir / title.lower().replace(" ", "_").replace("/", "-"))
        / filename
    )
    fig_single.savefig(output_single_plot_file)
    plt.close(fig_single)

    # then do the same and plot this in the main figure
    plot_func(ax, **kwargs)
    if axhline_wrapper is not None:
        axhline_wrapper(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def colorbar(
    im: cm.ScalarMappable, *, fig: plt.Figure, label: str = const.CBAR_LABEL, **kwargs
) -> Colorbar:
    cbar = fig.colorbar(im, **kwargs)
    cbar.set_label(
        label=label, fontsize=mpl_constants.MPL_FONT_SIZE_DEFAULT["axes.labelsize"]
    )
    cbar.ax.tick_params(
        labelsize=mpl_constants.MPL_FONT_SIZE_DEFAULT["xtick.labelsize"]
    )


def histogram(
    ax: axes.Axes,
    data: np.ndarray,
    bins: int = const.HIST_NUM_BINS,
    *,
    show_mean: bool = False,
    show_var: bool = False,
) -> axes.Axes:
    """Wrapper to plot histogram.

    Args:
        ax (axes.Axes): Axis of a subplot.
        data (np.ndarray): Data for which the histrogram is created.
        bins (int, optional): Number of bins in histogram. Defaults to const.HIST_NUM_BINS.

    Returns:
        axes.Axes: The axes object on which was plotted.
    """
    ax.hist(data.flatten(), bins=bins, alpha=const.HIST_ALPHA)
    ax.set_ylabel("Quantity")
    ax.set_xlabel("Value")
    if show_mean or show_var:
        sigma_2 = data.var().item()
        mu = data.mean().item()
        if show_mean and show_var:
            label = rf"$\begin{{aligned}} \mu &\approx {mu:.3f} \\\sigma^2 &\approx {sigma_2:.2e} \end{{aligned}}$"
        elif show_mean:
            label = rf"$\begin{{aligned}} \mu \approx {mu:.3f}  \end{{aligned}}$"
        elif show_var:
            label = (
                rf"$\begin{{aligned}} \sigma^2 \approx {sigma_2:.2e} \end{{aligned}}$"
            )

        props = dict(boxstyle="round", facecolor="lightgrey", edgecolor="black")
        ax.text(
            0.95,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=mpl_constants.MPL_FONT_SIZE_DEFAULT["axes.labelsize"],
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )
    return ax


def imshow_colorbar(
    ax: axes.Axes,
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
) -> axes.Axes:
    vmin_old = vmin
    vmin = vmin or max(data.min(), 0)
    vmax = vmax or min(data.max(), const.THR_IMSHOW_MAX)

    if vmin_old is None and vmin > vmax:
        vmax = vmin + const.THR_IMSHOW_MAX
    else:
        assert vmin <= vmax, f"{vmin=}, {vmax=}"

    cmap = plt.get_cmap(const.CMAP).copy()
    cmap.set_over("red")
    cmap.set_under("blue")

    if data[data < vmin].any() and data[data > vmax].any():
        extend = "both"
    elif data[data < vmin].any():
        extend = "min"
    elif data[data > vmax].any():
        extend = "max"
    else:
        extend = None
    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
    im = ax.imshow(data, norm=norm, cmap=cmap)
    colorbar(
        im,
        fig=ax.figure,
        ax=ax,
        orientation="vertical",
        extend=extend,
        fraction=0.05,
        label=cbar_label or const.CBAR_LABEL,
    )
    ax.axis("off")
    return ax


def imsave_min_max(
    filepath: Pathlike, data: np.ndarray, threshold: float | None = None
) -> None:
    """Save an image with values above a threshold shown in red.

    Args:
        filepath (Pathlike): Path to save the image.
        data (np.ndarray): Image data.
        threshold (float): Threshold for marking high values. Defaults to 1.5.
    """
    threshold = threshold or const.THR_IMSHOW_MAX
    vmin = max(data.min(), 0)
    vmax = min(data.max(), threshold)

    if vmin > vmax:
        vmax = vmin + threshold

    cmap = plt.get_cmap(const.CMAP).copy()
    cmap.set_over("red")

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    normalized_data = norm(data)
    rgba_image = cmap(normalized_data)

    rgba_image[data > threshold, :-1] = [1.0, 0.0, 0.0]  # Set RGB to red
    rgba_image[data > threshold, -1] = 1.0  # Keep alpha channel as 1

    # # Apply normalization directly to the data

    # # Save the normalized image
    plt.imsave(
        filepath,
        rgba_image,
        cmap=cmap,
    )

    # Save the min and max values to a CSV file
    stats_file = filepath.with_suffix(".csv")  # Change file extension to .csv
    stats = {"Filename": [filepath.name], "Min": [data.min()], "Max": [data.max()]}
    df = pd.DataFrame(stats)
    df.to_csv(stats_file, index=False)


def plot_line_values(
    ax: axes.Axes,
    data: List[np.ndarray],
    line: int,
    labels: List[str],
    colors: List[str] = ["blue", "orange", "green"],
    alphas: List[float] | None = None,
    do_upper_lower: bool = False,
) -> axes.Axes:
    colors = [mpl_constants.MPL_COLORS[c] for c in colors]
    if alphas is None:
        if len(data) == 3:
            alphas = [0.5, 1.0, 0.8]
        else:
            alphas = [1.0] * len(data)
    for i in range(len(data)):
        ax.plot(data[i][line, :], color=colors[i], label=labels[i], alpha=alphas[i])
    ax.legend(loc="upper right")

    if do_upper_lower:
        ax.axhline(0, linestyle="--", color="black", alpha=0.5)
        ax.axhline(1, linestyle="--", color="black", alpha=0.5)
    return ax


def show_difference_map(
    ax: Any, fig: Figure, gt: np.ndarray, other: np.ndarray, title: str
) -> None:
    # always do some kind of thresholding
    dif = gt - other
    if (limit1 := max(dif.max(), abs(dif.min()))) <= (limit2 := 0.1 * gt.max()):
        limit = limit1
        do_thresholding = False
    else:
        limit = limit2
        do_thresholding = True

    cmap = plt.get_cmap("coolwarm").copy()
    if do_thresholding:
        cmap.set_under("purple")
        cmap.set_over("brown")
        extend = "both"
    else:
        extend = None

    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    im = ax.imshow(dif, cmap=cmap, norm=norm)
    ax.axis("off")

    # Add a colorbar to show the scale of the difference
    colorbar(
        im,
        fig=fig,
        ax=ax,
        label="Deviation / A.U.",
        orientation="vertical",
        extend=extend,
    )
    if isinstance(ax, axes.Axes):
        ax.set_title(title)
    else:
        ax.title(title)


def plot_single_img(
    img: np.ndarray,
    title: str | None = None,
    ax: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    vmax = vmax or img.max() + const.EPS
    vmin = vmin or img.min()
    cmap = plt.get_cmap(const.CMAP).copy()
    cmap.set_over("red")
    cmap.set_under("blue")

    if img[img < vmin].any() and img[img > vmax].any():
        extend = "both"
    elif img[img < vmin].any():
        extend = "min"
    elif img[img > vmax].any():
        extend = "max"
    else:
        extend = None
    norm = Normalize(vmin=vmin, vmax=vmax, clip=False)

    fig = plt.figure(figsize=get_figsize(1, 1))
    im = fig.imshow(img, cmap=cmap, norm=norm)
    fig.axis("off")
    if title is not None:
        fig.title(title)
    if ax is not None:  # if given, then show colorbar
        colorbar(
            im,
            fig=fig,
            ax=ax,
            orientation="vertical",
            extend=extend,
            label=const.CBAR_LABEL,
        )


def plot_grid(
    imgs: List[Tensorlike], grid_size: Tuple[int, int], filename: Pathlike
) -> None:
    """Plots a grid of images.

    Note:
        All images are clipped at `vmax = 1.5`.

    Args:
        imgs (List[Tensorlike]): Images to be plotted. Images must be 2D Arrays/Tensors. Also note, that
                    len(imgs) > prod(grid_size) is valid.
        grid_size (Tuple[int, int]): Grid size in HxW patches.
        filename (Pathlike): Name where the image will be stored.
    """
    fig, axs = plt.subplots(
        *grid_size, figsize=get_figsize(*grid_size), constrained_layout=True
    )
    for ax, img in zip(axs.flat, imgs):
        imshow_colorbar(ax, img, vmin=None, vmax=None)
    fig.savefig(filename)
    plt.close(fig)


def plot_ffc_process(
    data: Tensorlike,
    sff: Tensorlike,
    filename: Pathlike,
    data_title: str,
    *,
    show_stats: bool = False,
    show_mean: bool = False,
    show_var: bool = False,
) -> None:
    fig, axs = subplots(2, 2)
    ffc_holo = ffc.correct_flatfield(data, sff)
    imgs = [data, sff, ffc_holo]
    title_ffc = f"{data_title} / {const.ANNOT_MODEL_OUT}"
    titles = [data_title, const.ANNOT_MODEL_OUT, title_ffc]
    output_dir = Path(filename).parent
    filename = Path(filename).name
    for i in range(len(imgs)):
        subplot_and_store(
            axs.flat[i],
            imshow_colorbar,
            title=titles[i],
            output_dir=output_dir,
            filename=filename,
            data=imgs[i],
        )

    show_stats = show_stats or (show_mean and show_stats)
    show_mean = show_stats or show_mean
    show_var = show_stats or show_var

    def hist_wrapper(ax: axes.Axes, *args, **kwargs) -> axes.Axes:
        histogram(ax, *args, **kwargs)
        if show_mean or show_var:
            sigma_2 = ffc_holo.var().item()
            mu = ffc_holo.mean().item()
            if show_stats:
                label = rf"$\begin{{aligned}} \mu &\approx {mu:.3f} \\\sigma^2 &\approx {sigma_2:.2e} \end{{aligned}}$"
            elif show_mean:
                label = rf"$\begin{{aligned}} \mu \approx {mu:.3f}  \end{{aligned}}$"
            elif show_var:
                label = rf"$\begin{{aligned}} \sigma^2 \approx {sigma_2:.2e} \end{{aligned}}$"

            props = dict(boxstyle="round", facecolor="lightgrey", edgecolor="black")
            ax.text(
                0.95,
                0.95,
                label,
                transform=ax.transAxes,
                fontsize=mpl_constants.MPL_FONT_SIZE_DEFAULT["axes.labelsize"],
                verticalalignment="top",
                horizontalalignment="right",
                bbox=props,
            )
        return ax

    subplot_and_store(
        axs[1, 1],
        plot_func=hist_wrapper,
        title=f"Histogram ({title_ffc})",
        data=ffc_holo,
        output_dir=output_dir,
        filename=filename,
        xlabel="Value",
        ylabel="Quantity",
    )
    fig.savefig(output_dir / filename)
    plt.close(fig)
