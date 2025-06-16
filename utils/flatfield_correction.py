# standard libraries

# third party libraries
import torch
from torch import Tensor

# local packages
from utils.datatypes import Tensorlike

from utils.remove_outliers import remove_outliers
import utils.constants as const


__all__ = [
    "correct_flatfield",
    "set_lower_bound",
]


def correct_flatfield(
    image: Tensorlike,
    synthetic_flat_field: Tensorlike,
    *,
    log_space: bool = False,
    rm_outliers: bool = True
) -> Tensor:
    image = Tensor(image)
    synthetic_flat_field = set_lower_bound(
        Tensor(synthetic_flat_field), log_space=log_space
    )
    if not log_space:
        image = torch.log(image)
        synthetic_flat_field = torch.log(synthetic_flat_field)
    corrected_image = image - synthetic_flat_field
    corrected_image = set_lower_bound(corrected_image, log_space=True)
    if rm_outliers:
        if image.ndim == 4:
            temp_corrected_image = torch.zeros_like(corrected_image)
            for i, img in enumerate(corrected_image):
                temp_corrected_image[i] = remove_outliers(torch.exp(img))
            corrected_image = temp_corrected_image
        else:
            corrected_image = remove_outliers(torch.exp(corrected_image))
    else:
        corrected_image = torch.exp(corrected_image)
    return corrected_image


def set_lower_bound(image: Tensorlike, log_space: bool) -> Tensorlike:
    lower_bound = -20 if log_space else const.EPS
    image[image < lower_bound] = lower_bound
    return image
