# standard libraries
from typing import Dict, Tuple

# third party libraries

# local packages


__all__ = [
    "MPL_FONT_SIZE_DEFAULT",
    "LATEX_COLORS",
    "OTHER_LATEX_COLORS",
    "MPL_COLORS",
]


MPL_FONT_SIZE_DEFAULT: Dict[str, int] = {
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
}


MPL_COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "yellow": "#bcbd22",
    "cyan": "#17becf",
}

LATEX_COLORS = {
    "blue": "#0000FF",
    "red": "#FF0000",
    "orange": "#FFA500",
    "magenta": "#FF00FF",
    "cyan": "#00FFFF",
    "black": "#000000",
    "green": "#00FF00",  # "#4E9A06", # not actual latex?
    "purple": "#5C3566",
    "brown": "#8F5902",
}

OTHER_LATEX_COLORS = {
    # Basic LaTeX Colors
    "black": "#000000",
    "white": "#FFFFFF",
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "yellow": "#FFFF00",
    # Additional `xcolor` Colors
    "orange": "#FFA500",
    "violet": "#EE82EE",
    "purple": "#800080",
    "brown": "#A52A2A",
    "lime": "#BFFF00",
    "olive": "#808000",
    "pink": "#FFC0CB",
    "teal": "#008080",
    "navy": "#000080",
    "gray": "#808080",
    "lightgray": "#D3D3D3",
    "darkgray": "#A9A9A9",
}
