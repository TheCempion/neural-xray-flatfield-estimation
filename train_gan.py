# standard libraries
import argparse
import collections
from typing import List, Any, Tuple

# third party libraries
import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate

# local packages
import data_loader as module_data
import data_loader.datasets as module_datasets
import utils.data_normalization as module_norm
import model.loss as module_loss
import model.gans as module_gans
from base import BaseGAN
from parse_config import ConfigParser
from trainer import GANTrainer
from utils import prepare_device, init_dataloader
import utils.torch_settings as torch_settings
import utils.constants as const


__all__ = [
    "",
]


def main(config: ConfigParser):
    # fix random seeds for reproducibility
    seed = config.get("seed", const.SEED)
    torch_settings.set_reproducibility(seed)  # fix random seeds for reproducibility

    logger = config.get_logger("train")
    logger.info(f"Output Directory: {config.save_dir}\n\n")

    # setup data_loader instances
    # default collate_fn struggles with None values
    def collate_fn_to_hanle_Nones(
        batch: List[Tuple[Tensor, Tensor, Tensor | None]],
    ) -> Any:
        # logger.debug(batch)
        batch_data = default_collate([item[0] for item in batch])
        batch_target = default_collate([item[1] for item in batch])

        if any([item[2] is None for item in batch]):
            batch_condition = None
        else:
            batch_condition = default_collate([item[2] for item in batch])
        return batch_data, batch_target, batch_condition

    # intialize dataset and setup data_loader instances
    dataset = config.init_obj("dataset", module_datasets)
    data_loader_class = getattr(module_data, config["data_loader"]["type"])

    data_loader, valid_data_loader = init_dataloader(
        dataset=dataset,
        data_loader_class=data_loader_class,
        training=True,
        **config["data_loader"]["args"],
        collate_fn=collate_fn_to_hanle_Nones,
    )

    if config.get("valid_data_loader", None) is not None:
        dataset = config.init_obj("dataset_valid_dataloader", module_datasets)
        data_loader_class = getattr(module_data, config["valid_data_loader"]["type"])
        valid_data_loader_2, _ = init_dataloader(
            dataset=dataset,
            data_loader_class=data_loader_class,
            training=False,
            **config["valid_data_loader"]["args"],
            collate_fn=collate_fn_to_hanle_Nones,
        )
    elif (new_args := config.get("valid_data_loader_dataset_args", None)) is not None:
        # use same dataset with (possibly) updated parameters for the validation data loader
        valid_batch_size = new_args.pop(
            "batch_size", config["data_loader"]["args"]["batch_size_valid"]
        )
        valid_dataset_args = config["dataset"]["args"] | new_args
        valid_data_loader_2, _ = init_dataloader(
            dataset=getattr(module_datasets, config["dataset"]["type"])(
                **valid_dataset_args
            ),
            data_loader_class=data_loader_class,
            training=False,
            batch_size=valid_batch_size,
            collate_fn=collate_fn_to_hanle_Nones,
        )
    else:
        valid_data_loader_2 = None

    if valid_data_loader is None:
        valid_data_loader, valid_data_loader_2 = valid_data_loader_2, valid_data_loader

    logger.info(f"{dataset=}")
    logger.info(f"{data_loader=}")
    logger.info(f"{valid_data_loader=}")
    logger.info(f"{valid_data_loader_2=}")

    # intialize data normalization class
    if config.get("data_normalizer", None) is None:
        data_normalizer = module_norm.DummyNormalizer()
    else:
        data_normalizer = config.init_obj("data_normalizer", module_norm)
    logger.info(f"{data_normalizer=}")

    # build model architecture, then print to console
    gan: BaseGAN = config.init_obj("arch", module_gans)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device()
    gan = gan.to(device)
    if len(device_ids) > 1:
        gan = torch.nn.DataParallel(gan, device_ids=device_ids)

    # get function handles of loss
    loss = config.init_obj("loss", module_loss)
    logger.info(f"{loss=}")

    config["trainer"]["early_stop"] = config["trainer"].get("early_stop", "inf")

    # optim and lr_scheduler
    optimizerG, lr_schedulerG = config.init_optim_lr_scheduler(gan.netG, "G")
    optimizerD, lr_schedulerD = config.init_optim_lr_scheduler(gan.netD, "D")
    logger.info(f"{optimizerG=}")
    logger.info(f"{optimizerD=}")
    logger.info(f"{lr_schedulerG=}")
    logger.info(f"{lr_schedulerD=}")

    logger.info(gan)
    trainer = GANTrainer(
        gan=gan,
        loss=loss,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        config=config,
        device=device,
        data_loader=data_loader,
        data_normalizer=data_normalizer,
        valid_data_loader=valid_data_loader,
        lr_schedulerG=lr_schedulerG,
        lr_schedulerD=lr_schedulerD,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train Model")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None).",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all).",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
