# standard libraries
import argparse

# third party libraries

# local packages
from parse_config import ConfigParser
import evaluation as module_eval
from evaluation import BaseEvaluator
import utils.constants as const


def main(config: ConfigParser) -> None:
    evaluator_type = config["evalutation"].pop("type")
    model_evaluator: BaseEvaluator = getattr(module_eval, evaluator_type)(config)
    model_evaluator.do_inference()
    model_evaluator.create_stats()
    model_evaluator.create_plots()
    model_evaluator.logger.info("Done with evaluation.")


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-e",
        "--ext",
        default=const.FILE_EXT,
        type=str,
        help="optional file extension for the plots.",
    )
    config = ConfigParser.from_args(args, test=True)
    main(config)
