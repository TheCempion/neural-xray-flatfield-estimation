import logging
from typing import List

from holowizard_livereco_server.api.parameters import RecoParams
from holowizard_livereco_server.api.parameters import FlatfieldCorrectionParams
from holowizard_livereco_server.api.viewer import Viewer

from .find_focus_flatfieldcorrection import find_focus as find_focus_internal
from holowizard_livereco_server.api.functions.default_load_data_callback import (
    default_load_data_callback,
)


def find_focus(
    glob_data_path,
    flatfield_correction_params: FlatfieldCorrectionParams,
    reco_params: RecoParams,
    image_index,
    load_data_callback=default_load_data_callback,
    viewer: List[Viewer] = None,
):
    data_path_loaded, data = load_data_callback(glob_data_path, image_index)

    reco_params.measurements[0].data_path = data_path_loaded
    reco_params.measurements[0].data = data

    logging.image_debug("loaded", data)

    z01_guess, z01_values_history, loss_values_history = find_focus_internal(
        flatfield_correction_params, reco_params, viewer
    )

    return z01_guess, z01_values_history, loss_values_history
