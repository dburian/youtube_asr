from __future__ import annotations
from coolname import generate
from typing import TYPE_CHECKING, Literal

import datasets as hf_datasets
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from torch.utils.data import DataLoader

from youtube_asr.dataset import SR, load_dataset
from youtube_asr import preprocess
import lightning as L
from lightning.pytorch import loggers as pl_loggers
import os

if TYPE_CHECKING:
    from typing import Any

DEFAULT_LIGHTNING_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../logs")
)


def upsample_factors_to_equal_label_representation() -> dict[str, float]:
    """Returns upsample factors so that label distributions are more evened-out."""
    return {
        "Speech": 1.0838117106773824,
        "Music": 1.0,
        "Animal": 8.137931034482758,
        "Musical instrument": 12.586666666666666,
        "Vehicle": 6.779174147217235,
        "Singing": 14.142322097378278,
        "Water": 19.463917525773194,
        "Tools": 53.94285714285714,
        "Wind": 26.405594405594407,
        "Silence": 94.4,
        "Gunshot": 236.0,
    }


def get_mfcc_dataloader(
    split: Literal["train", "dev"],
    batch_size: int,
    shuffle: bool,
    mfcc_kwargs: dict[str, Any],
    time_wrap: float = 0.0,
    mask_frequency: float = 0.0,
    max_frequency_mask_range: int = 5,
    mask_time: float = 0.0,
    max_time_mask_range: int = 80,
    upsample_factors: dict[str, float] | None = None,
    norm_mfcc: bool = False,
    num_workers: int = 1,
) -> tuple[DataLoader, list[str]]:
    """Creates a DataLoader for classifying MFCCs.

    Parameters
    ----------
    split: str
        Either 'train' or 'dev'
    batch_size: int
    shuffle: bool
        Whether to shuffle dataset or not.
    mfcc_kwargs: dict[str, Any]
        Key word arguments passed to MFCC transformation. Refer to
        `torchaudio.transforms.MFCC`.
    time_wrap: float
        Probablity to apply `preprocess.time_wrap` to a sample.
    mask_frequency: float
        Probability to apply frequency masking.
    max_frequency_mask_range: int
        Maximum MFCCs to mask out during frequency masking
    mask_time: float
        Probability to apply time masking.
    max_time_mask_range: int
        Maximum frames to mask out during time masking.
    upsample_factors: dict[str, float] | None
        How much to duplicate examples with given labels. Factors of examples
        with multiple labels are averaged.
    norm_mfcc: bool
        Whether to normalize MFCCs to have zero mean and standard deviation of 1.
    num_workers: int
        Number of workers for DataLoader.

    Returns
    ------
    tuple[DataLoader, list[str]]
        DataLoader and a list of target names in the same order as they were
        used to generate bit flags (targets).
    """
    ds = load_dataset(split)

    target_names = sorted({label for label in ds.features["labels"]})
    preprocessor = preprocess.Preprocess(
        target_names=target_names,
        pad_to=SR * 10,
        mfcc_kwargs=mfcc_kwargs,
        time_wrap_prob=time_wrap,
        mask_frequency_prob=mask_frequency,
        max_frequency_mask_range=max_frequency_mask_range,
        mask_time_prob=mask_time,
        max_time_mask_range=max_time_mask_range,
        upsample_factors=upsample_factors,
        norm_mfcc=norm_mfcc,
    )

    ds = preprocessor.process(ds)

    ds = ds.rename_column("mfcc", "inputs")

    if shuffle:
        ds = ds.shuffle()

    return DataLoader(
        ds.with_format("torch"),
        batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=12,
    ), target_names


def get_mfcc_dataloaders(
    batch_size: int,
    n_mfcc: int = 64,
    log_mels: bool = True,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    time_wrap: float = 0.0,
    upsample_factors: dict[str, float] | None = None,
    mask_frequency: float = 0.0,
    max_frequency_mask_range: int = 5,
    mask_time: float = 0.0,
    max_time_mask_range: int = 80,
    num_workers: int = 1,
    norm_mfcc: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
    mfcc_kwargs = {
        "n_mfcc": n_mfcc,
        "log_mels": log_mels,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "n_mels": n_mels,
    }
    train, target_names = get_mfcc_dataloader(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        mfcc_kwargs=mfcc_kwargs,
        time_wrap=time_wrap,
        num_workers=num_workers,
        upsample_factors=upsample_factors,
        mask_frequency=mask_frequency,
        max_frequency_mask_range=max_frequency_mask_range,
        mask_time=mask_time,
        max_time_mask_range=max_time_mask_range,
        norm_mfcc=norm_mfcc,
    )
    val, val_target_names = get_mfcc_dataloader(
        split="dev",
        batch_size=batch_size,
        shuffle=False,
        mfcc_kwargs=mfcc_kwargs,
        num_workers=num_workers,
        norm_mfcc=norm_mfcc,
    )

    assert tuple(target_names) == tuple(val_target_names)

    return train, val, target_names


def get_trainer(
    limit_train_batches: int | float | None = None,
    limit_val_batches: int | float | None = None,
    max_epochs: int | None = None,
    check_val_every_n_epoch: int | None = 1,
    log_every_n_steps: int | None = None,
    seed: int = 42,
) -> L.Trainer:
    """Utility method to create lightning Trainer with my default settings."""
    training_id = "_".join(generate(2))

    L.seed_everything(seed)

    print("-" * 80)
    print(f"EXPERIMENT: {training_id}")
    print("-" * 80)
    pl_logger = pl_loggers.TensorBoardLogger(
        os.path.join(DEFAULT_LIGHTNING_ROOT_DIR, "logs", training_id),
        name=None,
    )
    return L.Trainer(
        default_root_dir=DEFAULT_LIGHTNING_ROOT_DIR,
        callbacks=[
            ModelSummary(depth=-1),
            ModelCheckpoint(
                os.path.join(DEFAULT_LIGHTNING_ROOT_DIR, "checkpoints", training_id),
                monitor="val/f1/macro",
                mode="max",
            ),
        ],
        logger=pl_logger,
        num_sanity_val_steps=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=log_every_n_steps,
    )
