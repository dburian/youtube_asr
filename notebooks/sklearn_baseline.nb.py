# %% [markdown]
# # Baseline classifier with sklearn
#
# - TODO:
#     - play with mfcc kwargs
#     - update interface to old neighbors
#     - improve New Neighbors
#     - try pca with new neightbors
#     - eval SVC on dev
#     - look at MFCCs -- why they are so different?
#     - eval with window defined for mfcc

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

from youtube_asr.dataset import load_dataset
from youtube_asr import preprocess
import pandas as pd

# %% [markdown]
# ## KNearestNeighbors
# %%
from typing import Any, NamedTuple


class PrepedKNNData(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    target_names: list[str]


def prep_knn_data(split: str, **mfcc_kwargs) -> PrepedKNNData:
    ds = load_dataset(split)
    ds = ds.to_iterable_dataset()
    ds = preprocess.pad_to(ds, 16_000 * 10)
    ds = preprocess.to_mfcc(ds, **mfcc_kwargs)

    ds = ds.with_format("np")

    all_labels = [label for label in ds.features["labels"]]
    y = np.stack(
        [[label_dict[label] for label in all_labels] for label_dict in ds["labels"]],
        axis=0,
    )

    X = ds["mfcc"]
    ds_len, channels, n_mfcc, frames = X.shape
    assert channels == 1
    X = X.reshape((ds_len, n_mfcc, frames))
    return PrepedKNNData(X, y, all_labels)


# %%
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm

# %%
from sklearn.base import BaseEstimator


class ChunkedKNeighborsClassifier(BaseEstimator):
    """KNN on chunks of mfccs"""

    def __init__(
        self, chunk_size: int, chunk_hop_length: int, knn_model: Any, **kwargs
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_hop_length = chunk_hop_length
        self.knn_model = knn_model

        self.knn_model.set_params(
            **{
                key[len("knn_model__") :]: value
                for key, value in kwargs.items()
                if key.startswith("knn_model__")
            }
        )

    def get_params(self, deep=True):
        return super().get_params(True)

    def _to_chunks(self, mfccs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample_idxs = []
        chunks = []
        for sample_idx, mfcc in tqdm(
            enumerate(mfccs), desc="Splitting to chunks", total=mfccs.shape[0]
        ):
            for chunk_begin in range(0, mfcc.shape[1], self.chunk_hop_length):
                chunk_begin = max(0, min(chunk_begin, mfcc.shape[1] - self.chunk_size))
                chunk = mfcc[:, chunk_begin : chunk_begin + self.chunk_size]
                filters, chunk_frames = chunk.shape
                if chunk_frames < self.chunk_size:
                    chunk = np.concatenate(
                        [
                            chunk,
                            np.zeros((filters, self.chunk_size - chunk_frames)),
                        ],
                        axis=-1,
                    )

                chunks.append(chunk.reshape((-1,)))
                sample_idxs.append(sample_idx)

        return np.stack(chunks, axis=0), np.array(sample_idxs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        chunks, sample_idxs = self._to_chunks(X)

        self._fit_sample_idxs = sample_idxs
        self._fit_chunk_labels = y[sample_idxs]
        self.knn_model.fit(chunks, self._fit_chunk_labels)

    def predict(self, X: np.ndarray) -> np.ndarray:
        chunks, sample_idxs = self._to_chunks(X)

        all_dists, all_fit_idxs = self.knn_model.kneighbors(chunks)

        preds = []
        for idx in tqdm(range(X.shape[0]), desc="Predictions"):
            this_sample = sample_idxs == idx
            dists = all_dists[this_sample].reshape((-1,))
            fit_idxs = all_fit_idxs[this_sample].reshape((-1,))

            dists_sort_idxs = np.argsort(dists)
            votes = []
            weights = []
            for sort_idx in dists_sort_idxs[: self.knn_model.n_neighbors]:
                dist, fit_idx = dists[sort_idx], fit_idxs[sort_idx]
                label = self._fit_chunk_labels[fit_idx]
                votes.append(label)
                weights.append(dist)

            weights = np.array(weights)
            votes = np.stack(votes, axis=0, dtype=np.float32)
            if (weights == 0).any():
                votes = votes[weights == 0]
            else:
                weights = 1 / weights
                votes *= weights.reshape((-1, 1))

            preds.append(votes.mean(axis=0).round().clip(min=0, max=1))

        return np.stack(preds, axis=0)


# %% [markdown]
# ### Overfit
# %%
mfcc_kwargs = dict(
    n_mfcc=13, n_fft=4 * 4096, win_length=4 * 4096, hop_length=256, n_mels=64
)

train_data = prep_knn_data("train", **mfcc_kwargs)

# %%
model = ChunkedKNeighborsClassifier(
    chunk_size=512, chunk_hop_length=64, knn_model=KNeighborsClassifier(n_neighbors=5)
)

# %%
model.fit(train_data.X, train_data.y)
# %%
y_pred = model.predict(train_data.X[:100])
# %%
print(
    classification_report(
        train_data.y[:100], y_pred, target_names=train_data.target_names
    )
)
# %% [markdown]
# ### Random search
# %%
from sklearn.metrics import f1_score


def macro_f1_scoring(estimator, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = estimator.predict(X)
    return f1_score(y, y_pred, average="macro")


# %%
from sklearn.model_selection import RandomizedSearchCV
from youtube_asr.dataset import MultilabelStratifiedGroupKFold, stratified_splits

half_train_idxs = next(stratified_splits(train_data.y, 2))

params = {
    "chunk_size": [256, 512],
    "chunk_hop_length": [64, 128],
    "knn_model__n_neighbors": [5, 12, 24, 32],
}

rand_search = RandomizedSearchCV(
    estimator=ChunkedKNeighborsClassifier(
        chunk_size=256, chunk_hop_length=64, knn_model=KNeighborsClassifier(n_jobs=10)
    ),
    param_distributions=params,
    n_iter=10,
    random_state=42,
    verbose=4,
    scoring=macro_f1_scoring,
    cv=MultilabelStratifiedGroupKFold(n_splits=5),
)
# %%
rand_search.fit(train_data.X[half_train_idxs], train_data.y[half_train_idxs])
# %%
pd.DataFrame(rand_search.cv_results_).sort_values("mean_test_score")


# %% [markdown]
# ### Best on dev
# %%
model = ChunkedKNeighborsClassifier(
    chunk_size=256, chunk_hop_length=128, knn_model=KNeighborsClassifier(n_neighbors=32)
)

# %%
model.fit(train_data.X, train_data.y)
# %%
dev_data = prep_knn_data("train", **mfcc_kwargs)

# %%
y_pred = model.predict(dev_data.X)

# %%
print(classification_report(dev_data.y, y_pred, target_names=dev_data.target_names))

# %% [markdown]
# ## Old ChunkedKNearestNeighbors
# %%
from typing import Any, NamedTuple
import torch


def to_chunks(
    sample_batch: dict[str, Any],
    chunk_size: int = 512,
    chunk_hop_length: int = 64,
) -> dict[str, Any]:
    batch_chunks = []
    batch_labels = []
    batch_audio_idxs = []
    for mfcc, labels, audio_idx in zip(
        sample_batch["mfcc"], sample_batch["labels"], sample_batch["audio_idx"]
    ):
        for chunk_begin in range(0, mfcc.shape[1], chunk_hop_length):
            chunk_begin = max(0, min(chunk_begin, mfcc.shape[1] - chunk_size))
            chunk = mfcc[:, chunk_begin : chunk_begin + chunk_size]
            filters, chunk_frames = chunk.shape
            if chunk_frames < chunk_size:
                chunk = torch.concat(
                    [
                        chunk,
                        torch.zeros((filters, chunk_size - chunk_frames)),
                    ],
                    axis=-1,
                )
            batch_chunks.append(chunk)
            batch_labels.append(labels)
            batch_audio_idxs.append(audio_idx)

    return {
        "labels": batch_labels,
        "chunk": batch_chunks,
        "audio_idx": batch_audio_idxs,
    }


class PrepedChunkData(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    audio_idxs: np.ndarray
    target_names: list[str]


# TODO: Doesn't actually use chunk args
def prep_chunk_data(
    split: str, chunk_size: int, chunk_hop_length: int, **mfcc_kwargs
) -> PrepedChunkData:
    ds = load_dataset(split)
    preprocessor = Preprocess(num_proc=1)
    ds = preprocessor.to_mfcc(ds, **mfcc_kwargs)

    ds = ds.map(
        to_chunks,
        remove_columns=ds.column_names,
        batch_size=256,
        batched=True,
    )

    ds = ds.with_format("np")

    all_labels = [label for label in ds.features["labels"]]
    y = np.stack(
        [[label_dict[label] for label in all_labels] for label_dict in ds["labels"]],
        axis=0,
    )

    X = ds["chunk"].reshape((len(ds), -1))
    return PrepedChunkData(X, y, ds["audio_idx"], all_labels)


# %%
# %%

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm


class ChunkScorer:
    def __init__(
        self,
        ind_to_audio_idx: np.ndarray,
        target_names: list[str],
        return_report: bool = True,
    ) -> None:
        self.audio_idx_to_ind = {}
        unique_audio_idx = np.unique(ind_to_audio_idx)
        for audio_idx in unique_audio_idx:
            self.audio_idx_to_ind[audio_idx] = np.nonzero(
                ind_to_audio_idx == audio_idx
            )[0]

        self.expecting_len = len(ind_to_audio_idx)
        self.target_names = target_names
        self.return_report = return_report

    def nearest_neighbor(
        self, classifier, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, dict[str, Any]] | float:
        assert (
            self.expecting_len == X.shape[0]
        ), "Given differently-sized indices to audio_idxs than prediction data."

        for audio_idx in tqdm(
            sorted(self.audio_idx_to_ind), desc="Aggregating predictions over chunks"
        ):
            chunk_idxs = self.audio_idx_to_ind[audio_idx]
            chunk_feats = X[chunk_idxs]

            neighbors = self.asdf

    def sum_probas(
        self,
        classifier,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, dict[str, Any]] | float:
        assert (
            self.expecting_len == X.shape[0]
        ), "Given differently-sized indices to audio_idxs than prediction data."

        print("Predicting probabilities")
        probas = np.array(classifier.predict_proba(X))

        preds = []
        trues = []
        for audio_idx in tqdm(
            sorted(self.audio_idx_to_ind), desc="Aggregating predictions over chunks"
        ):
            chunk_idxs = self.audio_idx_to_ind[audio_idx]
            audio_trues = y[chunk_idxs]
            audio_preds = probas[:, chunk_idxs, 1]

            assert (
                audio_trues == audio_trues[0]
            ).all(), f"Labels are not the same for all chunks of audio {audio_idx}: {chunk_idxs}, {audio_trues}"
            audio_trues = audio_trues[0]

            audio_preds = audio_preds.sum(axis=1).round().clip(min=0, max=1)
            preds.append(audio_preds)
            trues.append(audio_trues)

        macro_f1 = f1_score(trues, preds, average="macro")

        if not self.return_report:
            return macro_f1

        return macro_f1, classification_report(
            preds, trues, target_names=self.target_names, output_dict=True
        )


# %%
mfcc_kwargs = dict(n_mfcc=13, n_fft=4 * 4096, hop_length=256, n_mels=64)
chunk_size = 512
chunk_hop_length = 64

# %%
train_data = prep_chunk_data(
    split="train",
    chunk_size=chunk_size,
    chunk_hop_length=chunk_hop_length,
    **mfcc_kwargs,
)
# %%
model = KNeighborsClassifier(n_neighbors=32, weights="distance", n_jobs=-1)

# %%
model.fit(train_data.X, train_data.y)

# %%
dev_data = prep_chunk_data(
    "dev",
    chunk_hop_length=chunk_hop_length,
    chunk_size=chunk_size,
    **mfcc_kwargs,
)

# %%
dev_scorer = ChunkScorer(dev_data.audio_idxs, dev_data.target_names)

# %%
acc, rep = dev_scorer(model, dev_data.X, dev_data.y)

# %%
import seaborn as sns
import pandas as pd

# %%
sns.heatmap(pd.DataFrame(rep).iloc[:-1, :].T, annot=True)

# %% [markdown]
# ### Randomized search w/o stratified folds

# %%
from sklearn.model_selection import KFold, RandomizedSearchCV

# %%


cv = KFold(5, shuffle=True, random_state=42)

for train_idxs, test_idxs in cv.split(train_data.X):
    X_train, X_test = train_data.X[train_idxs], train_data.X[test_idxs]
    y_train, y_test = train_data.y[train_idxs], train_data.y[test_idxs]


# %%
gs = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions={
        "n_neighbors": [3, 5, 12, 32, 64],
        "weights": ["uniform", "distance"],
    },
    n_iter=8,
    verbose=4,
    cv=5,
)


# %%
gs.fit(train_data.X, train_data.y)

# %% [markdown]
# ## Random Forest
# %% [markdown]
# ### Overfit

# %%

from sklearn.decomposition import PCA


class PrepedRFData(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    target_names: list[str]
    pca: PCA | None


def prep_rf_data(
    split: str, pca_components: int | None = None, **mfcc_kwargs
) -> PrepedRFData:
    ds = load_dataset(split)
    preprocessor = Preprocess(num_proc=1)
    ds = preprocessor.pad_to(ds, 16_000 * 10)
    ds = preprocessor.to_mfcc(ds, **mfcc_kwargs)

    ds = ds.with_format("np")

    all_labels = [label for label in ds.features["labels"]]
    y = np.stack(
        [[label_dict[label] for label in all_labels] for label_dict in ds["labels"]],
        axis=0,
    )

    X = ds["mfcc"].reshape((len(ds), -1))
    pca = None
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    return PrepedRFData(X=X, y=y, target_names=all_labels, pca=pca)


# %%
from sklearn.ensemble import RandomForestClassifier

# %%
train_data = prep_rf_data("train", **mfcc_kwargs)

# %%
rf = RandomForestClassifier(verbose=3)

# %%
rf.fit(train_data.X, train_data.y)
# %%
y_pred = rf.predict(train_data.X)

print(classification_report(train_data.y, y_pred, target_names=train_data.target_names))
# %% [markdown]
# ### Random search

# %%
params = {
    "n_estimators": [100, 250],
    "max_depth": [6, 15, 25],
    "min_samples_leaf": [6, 8, 12],
}


rand_search = RandomizedSearchCV(
    RandomForestClassifier(verbose=3, random_state=42),
    param_distributions=params,
    n_iter=12,
    scoring=macro_f1_scoring,
    verbose=3,
    n_jobs=6,
    random_state=42,
)
# %%
rand_search.fit(train_data.X, train_data.y)

# %%
all_cv_results = pd.DataFrame(rand_search.cv_results_)

# %%
all_cv_results.sort_values("mean_test_score")

# %% [markdown]
# #### With PCA

# %%
params = {
    "n_estimators": [100, 250],
    "max_depth": [6, 15, 25],
    "min_samples_leaf": [6, 8, 12],
}


rand_search = RandomizedSearchCV(
    RandomForestClassifier(verbose=3, random_state=42),
    param_distributions=params,
    n_iter=12,
    scoring=macro_f1_scoring,
    verbose=3,
    n_jobs=6,
    random_state=42,
)
# %%
train_data_pca = prep_rf_data("train", pca_components=0.9, **mfcc_kwargs)

# %%
rand_search.fit(train_data_pca.X, train_data_pca.y)

# %%
pca_cv_results = pd.DataFrame(rand_search.cv_results_)
pca_cv_results["param_pca_components"] = 0.9

# %%
all_cv_results = pd.concat([all_cv_results, pca_cv_results], axis=0)

# %%
all_cv_results.sort_values("mean_test_score")

# %% [markdown]
# ### Best on dev

# %%
rf = RandomForestClassifier(
    n_estimators=250,
    min_samples_leaf=12,
    max_depth=25,
    random_state=42,
    verbose=3,
)

rf.fit(train_data.X, train_data.y)


# %%
dev_data = prep_rf_data("dev", **mfcc_kwargs)

y_pred = rf.predict(dev_data.X)

print(classification_report(dev_data.y, y_pred, target_names=dev_data.target_names))

# %% [markdown]
# ## SVM
# %%
from itertools import product, zip_longest

svc_params = {
    "C": [1.0, 1.5, 0.8],
    "gamma": ["scale", "auto"],
    "kernel": ["rbf", "linear"],
}

[
    dict(sth)
    for sth in product(
        *[zip_longest([key], svc_params[key], fillvalue=key) for key in svc_params]
    )
]

# %%
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from itertools import product, zip_longest

params = {
    "estimator__C": [1.0, 1.5, 0.8],
    "estimator__gamma": ["scale", "auto"],
    "estimator__kernel": ["rbf", "linear"],
}

rand_search = RandomizedSearchCV(
    OneVsRestClassifier(estimator=SVC(random_state=42, verbose=True, max_iter=20000)),
    param_distributions=params,
    n_iter=9,
    scoring=macro_f1_scoring,
    verbose=3,
    n_jobs=6,
    random_state=42,
    cv=MultilabelStratifiedGroupKFold(n_splits=5),
)
# %%
train_data_pca.X.shape

# %%
rand_search.fit(train_data_pca.X, train_data_pca.y)

# %%
pd.DataFrame(rand_search.cv_results_).sort_values("mean_test_score")

# %%
dev_data_pca = ...
