from __future__ import annotations
from copy import copy
import logging
from typing import TYPE_CHECKING, Generator
from sklearn.model_selection import StratifiedGroupKFold

import numpy as np
import os
import torchaudio
import datasets as hf_datasets
import pandas as pd

if TYPE_CHECKING:
    from typing import Any

SR = 16_000

logger = logging.getLogger(__name__)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/"))


def _get_tsv_path(split: str) -> str:
    return os.path.join(DATA_PATH, f"dataset_{split}.tsv")


def load_pandas(split: str) -> pd.DataFrame:
    df = pd.read_csv(_get_tsv_path(split), sep="\t")
    df["labels"] = df["labels"].apply(lambda labels_str: tuple(labels_str.split(",")))

    return df


def load_dataset(split: str) -> hf_datasets.Dataset:
    """Loads given split from hard-coded data directory into HF dataset."""
    ds = hf_datasets.Dataset.from_csv(_get_tsv_path(split), sep="\t")
    assert isinstance(ds, hf_datasets.Dataset)

    all_labels = {
        label for label_comb in ds.unique("labels") for label in label_comb.split(",")
    }

    def _separate_labels(sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "labels": {label: int(label in sample["labels"]) for label in all_labels}
        }

    ds = ds.map(_separate_labels)

    audio_dir = os.path.join(DATA_PATH, "audio")

    def _load_audio(sample_batch: dict[str, Any]) -> dict[str, Any]:
        wavs = []
        for filename in sample_batch["audio"]:
            wav, sr = torchaudio.load(os.path.join(audio_dir, f"{filename}.wav"))
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            wavs.append(wav)

        return {"wav": wavs}

    return ds.map(
        _load_audio,
        batched=True,
        batch_size=24,
    )


def stratified_splits(labels: np.ndarray, n_splits: int = 2) -> Generator[np.ndarray]:
    """Generates `n_splits` stratified according to label presence.

    Note that this is not the same as stratifying according to label
    combination. Each audio is duplicated for each label it has.
    StratifiedGroupKFold is then used to stratify the duplicated audios, where
    groups group together duplicates of the same audio. This ensures that no
    audio is in two splits at the same time.
    """
    labels = labels.copy() * np.arange(1, labels.shape[1] + 1).reshape((1, -1))
    labels_df = pd.DataFrame({"labels": [label for label in labels]}).explode("labels")

    labels_df = labels_df[labels_df["labels"] != 0]
    labels_df["labels"] = labels_df["labels"].astype(int)
    labels_df = labels_df.reset_index().rename(columns={"index": "group"})

    folds_gen = StratifiedGroupKFold(n_splits=n_splits)
    y = labels_df["labels"].to_numpy()
    groups = labels_df["group"].to_numpy()

    used_idxs = set()
    for _, fold_idxs in folds_gen.split(groups, y=y, groups=groups):
        unique_fold_idxs = np.unique(groups[fold_idxs])
        unique_fold_idxs_set = set(unique_fold_idxs)
        assert len(used_idxs & unique_fold_idxs_set) == 0
        used_idxs |= unique_fold_idxs_set

        yield unique_fold_idxs


class MultilabelStratifiedGroupKFold:
    """sklearn interface for cross-validation according with multilabel stratification.

    For more info see `youtube_asr.dataset.stratified_splits`.
    """

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def get_n_splits(
        self, X: np.ndarray, y: np.ndarray | None, groups: np.ndarray | None
    ) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        assert y is not None
        folds_idxs = stratified_splits(y, n_splits=self.n_splits)

        all_idxs = set(range(X.shape[0]))
        for test_idxs in folds_idxs:
            train_idxs = np.array(list(all_idxs - set(test_idxs)))
            yield train_idxs, test_idxs
