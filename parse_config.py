# standard libraries
import os
from typing import Dict, Any, Literal, Tuple
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
import glob
import shutil

# third party libraries
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch

# local packages
from logger import setup_logging
import utils.fileIO as fileIO
import utils.constants as const
import model.lr_scheduler as module_lr_scheduler
import utils.plotting as plotting


__all__ = [
    "ConfigParser",
]


class ConfigParser:
    def __init__(
        self,
        config: Dict[str, Any],
        modification: Dict[str, Any] | None = None,
        test: bool = False,
        cfg_fname: Path = None,
    ):
        """Parse json-configuration file.

        Handles hyperparameters for training, initializations of modules, checkpoint saving and logging module.

        Args:
            config (Dict[str, Any]): Dict containing configurations, hyperparameters for training.
            modification (Dict[str, Amy], optional): Keychain:value. specifying position values to be replaced from
                        config dict.
            test (bool, optional): Flag, if the current run is a training or test run. Deafaults to False.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.test = test
        self.cfg_fname = cfg_fname
        self._setup_logger()
        self._init_matplotlib()

    @classmethod
    def from_args(cls, args, options="", test: bool = False):
        """Initialize this class from some cli arguments. Used in train, test."""
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)

        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        assert (
            args.config is not None
        ), "Configuration file need to be specified. E.g., add '-c config.json'"
        cfg_fname = Path(args.config)
        config = fileIO.read_json(cfg_fname)
        if test:
            const.set_file_ext(args.ext)

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }
        return cls(config, modification, test=test, cfg_fname=cfg_fname)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self.config[key]
        except KeyError:
            return default

    def _setup_logger(self) -> None:
        # set save_dir where trained model and log will be saved.
        # AND check whether training was interrupted
        exper_name = self.config.get("name", "")
        mode = "test" if self.test else "train"
        cfg_name = f"cfg_{self.cfg_fname.stem}"
        sub_path = "/".join(
            str(self.cfg_fname.parent).split("/")[3:]
        )  # remove hpc/configs/<mode>
        save_dir_base = Path("hpc/output") / mode / exper_name / sub_path / cfg_name
        self.save_dir = save_dir_base

        # check, whether model was already trained and was preempted
        def clean_output_folder():
            # remove existing stored images and checkpoints
            for sub_path in [
                "plots/A_inputs",
                "plots/B_outputs",
                "plots/C_output_single",
                "plots/D_inputs_vs_output_with_line",
                "plots/E_target_vs_output",
                "plots/F_in_vs_target_vs_out",
                "plots/G__in_vs_target_vs_out_minmax",
                "plots/H_zoomed_in",
                "plots/I_FFC_holo",
                "plots/J_FFC_target",
                "plots/Z_target_with_line",
                "checkpoints",
            ]:
                if (path := (self.save_dir / sub_path)).is_dir():
                    shutil.rmtree(path)

        if self.save_dir.is_dir():
            if (self.save_dir / "model_last.pth").is_file():
                if not self.config.get("ignore_trained", False):
                    raise RuntimeError(
                        "Model was already trained. Need to set `ignore_trained` flag in config to override."
                    )
                else:
                    clean_output_folder()
            else:  # continue training, use last_checkpoint as starting point, elif model_best, else no loading
                if (self.save_dir / "checkpoints").is_dir() and (
                    (
                        checkpoints := [
                            p
                            for p in (self.save_dir / "checkpoints").rglob(
                                "checkpoint-*.pth"
                            )
                        ]
                    )
                    != []
                ):
                    sorted_checkpoints = sorted(
                        checkpoints, key=lambda p: int(p.stem.split("-")[-1])
                    )
                    self.resume = sorted_checkpoints[-1]
                    self.resume_epoch = int(self.resume.stem.split("-")[-1])

                elif (model_best_path := (self.save_dir / "model_best.pth")).is_file():
                    self.resume = model_best_path
                    self.resume_epoch = torch.load(model_best_path).get("epoch", None)
                else:
                    self.resume = None
                    self.resume_epoch = None

                # if self.resume is not None and not self.config.get("continue_training", False):
                #     print(
                #         "Will not continue training. Need to set `continue_training` flag in config \
                #             or delete outputs from previous training manually. Overriding existing stuff"
                #     )
                #     clean_output_folder()

        else:
            self.save_dir.mkdir(parents=False, exist_ok=False)

        # save updated config file to the checkpoint dir
        fileIO.write_json(self.config, self.save_dir / "config.json")

        # delete old tensorboard event files
        for tensorboard_file in list(glob.glob(str(self.save_dir / "events.out.*"))):
            Path(tensorboard_file).unlink()

        # configure logging module
        setup_logging(self.save_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def get_logger(self, name: str, verbosity: int = 2) -> logging.Logger:
        msg_verbosity = f"verbosity option {verbosity} is invalid. Valid options are {self.log_levels.keys()}."
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def _init_matplotlib(self) -> None:
        if const.MPL_USE_TEX:
            plotting.init_matplotlib()

    def init_optim_lr_scheduler(
        self, net: nn.Module, net_suffix: Literal["G", "D", ""] = ""
    ) -> Tuple[Optimizer, LRScheduler | None]:
        optim_path = self.config[f"optimizer{net_suffix}"]["args"].pop(
            "optim_path", None
        )
        trainable_params = filter(lambda p: p.requires_grad, net.parameters())
        optimizer: Optimizer = self.init_obj(
            f"optimizer{net_suffix}", torch.optim, trainable_params
        )

        if optim_path is not None:  # TODO: elif?
            optimizer.load_state_dict(torch.load(optim_path))

        # init LR Scheduler
        for m in [module_lr_scheduler, torch.optim.lr_scheduler]:
            try:
                scheduler = self.init_obj(f"lr_scheduler{net_suffix}", m, optimizer)
            except:
                scheduler = None
            else:
                break

        return optimizer, scheduler

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    @save_dir.setter
    def save_dir(self, val: Path):
        try:
            self._save_dir
        except AttributeError:
            self._save_dir = val
        else:
            raise ValueError("Cannot change save directory after it was set.")


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
