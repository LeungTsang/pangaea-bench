import os as os
import pathlib
import pprint
import random
import time
import wandb

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.datasets.base import RawGeoFMDataset, GeoFMDataset
from pangaea.engine.data_preprocessor import Preprocessor
from pangaea.engine.evaluator import Evaluator
from pangaea.engine.trainer import Trainer
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.logger import init_logger
from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_generator,
    seed_worker,
)


def get_exp_name(hydra_config: HydraConf) -> str:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment name.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    return f"{timestamp}-{fm}-{decoder}-{ds}"


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    # distributed training variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")

    use_wandb_this_rank = cfg.use_wandb and rank == 0

    # true if training else false
    train_run = cfg.train
    if train_run:
        exp_name = get_exp_name(HydraConfig.get())
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        # init wandb
        if use_wandb_this_rank:
            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
            )
            cfg["wandb_run_id"] = wandb.run.id
        OmegaConf.save(cfg, config_log_dir / "config.yaml")

    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"
        # load training config
        cfg_path = exp_dir / "configs" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
        if use_wandb_this_rank:

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="pangaea-bench",
                name=exp_name,
                config=wandb_cfg,
                id=cfg.get("wandb_run_id"),
                resume="allow",
            )

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    train_preprocessor = instantiate(cfg.preprocessing.train, dataset_cfg=cfg.dataset, encoder_cfg=cfg.encoder, _recursive_=False)
    val_preprocessor = instantiate(cfg.preprocessing.val, dataset_cfg=cfg.dataset, encoder_cfg=cfg.encoder, _recursive_=False)

    # get datasets
    raw_train_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="train")
    raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")
    train_dataset = GeoFMDataset(raw_train_dataset, train_preprocessor)
    val_dataset = GeoFMDataset(raw_val_dataset, val_preprocessor)
    logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    logger.info("Built {}.".format(encoder.model_name))

    # prepare the decoder (segmentation/regression)
    decoder: Decoder = instantiate(
        cfg.decoder,
        encoder=encoder,
    )
    decoder.to(device)
    decoder = torch.nn.parallel.DistributedDataParallel(
        decoder, device_ids=[local_rank], output_device=local_rank,
    )
    logger.info(
        "Built {} for with {} encoder.".format(
            decoder.module.model_name, type(encoder).__name__
        )
    )

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    # training
    if train_run:
        if 0 < cfg.limited_label < 1:
            n_train_samples = len(train_dataset)
            indices = random.sample(
                range(n_train_samples), int(n_train_samples * cfg.limited_label)
            )
            train_dataset = Subset(train_dataset, indices)
            logger.info(
                f"Created a subset of the train dataset, with {cfg.limited_label * 100}% of the labels available"
            )
        else:
            logger.info("The entire train dataset will be used.")

        # get train val data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            # persistent_workers=True causes memory leak
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=get_generator(cfg.seed),
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=False,
            collate_fn=collate_fn,
        )

        criterion = instantiate(cfg.criterion)
        optimizer = instantiate(cfg.optimizer, params=decoder.parameters())
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            total_iters=len(train_loader) * cfg.task.trainer.n_epochs,
        )

        val_evaluator: Evaluator = instantiate(
            cfg.task.evaluator, val_loader=val_loader, exp_dir=exp_dir, device=device, use_wandb=use_wandb_this_rank
        )
        trainer: Trainer = instantiate(
            cfg.task.trainer,
            model=decoder,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=val_evaluator,
            exp_dir=exp_dir,
            device=device,
            use_wandb=use_wandb_this_rank
        )
        # resume training if model_checkpoint is provided
        if cfg.ckpt_dir is not None:
            trainer.load_model(cfg.resume_from)

        trainer.train()

    # Evaluation
    test_preprocessor = instantiate(cfg.preprocessing.test, dataset_cfg=cfg.dataset, encoder_cfg=cfg.encoder, _recursive_=False)

    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    test_loader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset),
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_evaluator: Evaluator = instantiate(
        cfg.task.evaluator, val_loader=test_loader, exp_dir=exp_dir, device=device, use_wandb=use_wandb_this_rank
    )
    best_model_ckpt_path = get_best_model_ckpt_path(exp_dir)
    test_evaluator.evaluate(decoder, "best_model", best_model_ckpt_path)

    if use_wandb_this_rank:
        wandb.finish()


if __name__ == "__main__":
    main()
