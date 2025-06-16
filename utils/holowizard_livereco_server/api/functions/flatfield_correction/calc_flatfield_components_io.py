import logging
import pickle

from livereco.core.utils.fileio import load_multi_img_data
from livereco.core.preprocessing import (
    calculate_flatfield_components,
)

from holowizard_livereco_server.api.parameters import FlatfieldComponentsParams


def log_empties(empties):
    i = 0
    for empty in empties:
        logging.image_debug("calculate_flatfield_components_input_" + str(i), empty)
        i = i + 1


def calculate_flatfield_components(
    flatfield_components_params: FlatfieldComponentsParams,
):
    logging.debug("Load data")
    empties = load_multi_img_data(flatfield_components_params.measurements)
    log_empties(empties)

    logging.debug("Calculate components")
    logging.info(
        "Using " + str(flatfield_components_params.num_components) + " components"
    )
    components_model = calculate_flatfield_components(
        empties, flatfield_components_params.num_components
    )

    logging.debug(
        "Write components to file system: " + flatfield_components_params.save_path
    )

    with open(flatfield_components_params.save_path, "wb") as file:
        pickle.dump(components_model, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    logging.info("calculate_flatfield_components Finished")

    return components_model
