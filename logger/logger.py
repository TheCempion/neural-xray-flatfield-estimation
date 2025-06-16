# standard lilbraries
import logging
import logging.config
from pathlib import Path

# third party libraries

# local packages
from utils.datatypes import Pathlike
import utils.fileIO as fileIO


__all__ = [
    "setup_logging",
]


def setup_logging(
    save_dir: Path,
    log_config: Pathlike = "logger/logger_config.json",
    default_level: int = logging.INFO,
) -> None:
    """Setup logging configuration."""
    log_config = Path(log_config)
    if log_config.is_file():
        config = fileIO.read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)
