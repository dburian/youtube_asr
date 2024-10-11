# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
from typing import NamedTuple
from youtube_asr.dataset import load_dataset
from youtube_asr.train import get_mfcc_dataloaders
from youtube_asr.model import CNNClassifier, ClassifierLightningModule
import torch

# %%
torch.cuda.is_available()

# %%
train, val, _ = get_mfcc_dataloaders(
    batch_size=8, hop_length=256, log_mels=True, n_fft=1024, n_mfcc=13, win_length=1024
)

# %%
train

# %%
from tqdm.auto import tqdm

for _ in tqdm(train):
    pass

# %%
import lightning as L

# %%
trainer = L.Trainer(default_root_dir="./logs")

# %%
model = ClassifierLightningModule(classifier=CNNClassifier(11))

# %%
trainer.fit(model, train_dataloaders=train, val_dataloaders=val)

# %% [markdown]
# ## Testing layer sizes

# %%
train, _, _ = get_mfcc_dataloaders(batch_size=8, n_mfcc=13)

# %%
sample_batch = next(iter(train))
print(sample_batch.keys())
print(sample_batch["inputs"].shape)

# %%
in_dimension = sample_batch["inputs"].shape[2:]
in_dimension


# %%
class CNNLayer(NamedTuple):
    kernel: int | tuple[int, int]
    stride: int | tuple[int, int]
    pad: int


def maybe_2d(val: int | tuple[int, int], dim: int) -> int:
    if isinstance(val, tuple):
        return val[dim]
    return val


def adjust_dim(in_dimension: tuple[int, int], layer: CNNLayer) -> tuple[int, int]:
    def _adjust_single(w: int, k: int, p: int, s: int):
        return (w - k + 2 * p) // s + 1

    return tuple(
        _adjust_single(
            in_dimension[dim],
            maybe_2d(layer.kernel, dim),
            layer.pad,
            maybe_2d(layer.stride, dim),
        )
        for dim in range(2)
    )


def compute_features(
    layers: list[CNNLayer], in_dimension: tuple[int, int]
) -> list[tuple[int, int]]:
    print(f"Layer -1: {in_dimension}")
    dims = [in_dimension]
    for i, layer in enumerate(layers):
        in_dimension = adjust_dim(in_dimension, layer)
        print(f"Layer {i}: {in_dimension}")
        dims.append(in_dimension)

    print(f"Flatten: {dims[-1][0] * dims[-1][1]}")

    return dims


# %%
layers = [
    CNNLayer((3, 7), (1, 5), 0),
    CNNLayer((3, 5), (1, 3), 0),
    CNNLayer(3, 1, 0),
    CNNLayer(3, 1, 0),
]

compute_features(layers, in_dimension)

# %%
from youtube_asr.model import CNNClassifier

# %%
model = CNNClassifier(11)

# %%
for name, param in model.named_parameters():
    print(name, param.shape)
# %% [markdown]
# ## Testing my metric

# %%
from typing import Generator


def random_logits_iter(
    num_batches: int, batch_size: int, num_classes: int = 11
) -> Generator[torch.Tensor, None, None]:
    for _ in range(num_batches):
        yield torch.rand((batch_size, num_classes)) * torch.randint(-1024, 1024, (1,))


def random_targets_iter(
    num_batches: int, batch_size: int, num_classes: int = 11
) -> Generator[torch.Tensor, None, None]:
    for _ in range(num_batches):
        yield (torch.rand((batch_size, num_classes)) >= 0.5).to(torch.float32)


# %%

bs = 8
num_batches = 24

logits_iter = random_logits_iter(num_batches, bs)
targets_iter = random_targets_iter(num_batches, bs)


target_names = [str(i) for i in range(11)]
from youtube_asr.model import MultiLabelMetric

metric = MultiLabelMetric(target_names)
# %%
all_logits = []
all_targets = []
for logits, targets in zip(logits_iter, targets_iter):
    all_logits.append(
        (torch.nn.functional.sigmoid(logits) > 0.5).numpy().astype(np.int32)
    )
    all_targets.append(targets.numpy())
    metric.update(logits, targets)

all_targets = np.vstack(all_targets).astype(np.int32)
all_logits = np.vstack(all_logits)

all_targets.shape, all_targets.dtype, all_logits.shape, all_logits.dtype

# %%
from sklearn.metrics import classification_report
import numpy as np

metric_dict = metric.compute()
cls_report = classification_report(
    all_targets, all_logits, target_names=target_names, output_dict=True
)
# %%
# assert supports

pprint({key: value for key, value in metric_dict.items() if key.endswith("support")})
pprint(cls_report)

# %%
metric = "f1"
pprint({key: value for key, value in metric_dict.items() if key.endswith(metric)})
pprint({f"{key}_{metric}": value["f1-score"] for key, value in cls_report.items()})

# %%
from pprint import pprint

pprint(metric_dict)
# %%
pprint(cls_report)

# %% [markdown]
# ## Testing concatenation of dataset

# %%
from datasets import concatenate_datasets
from tqdm.auto import tqdm
import youtube_asr.preprocess as preprocess

# %%
from youtube_asr.dataset import load_dataset

# %%
ds = load_dataset("train")

# %%
ds = concatenate_datasets([ds for _ in range(4)], axis=0)

# %%
ds

# %%
ds_iter = ds.to_iterable_dataset(num_shards=4)

# %%
ds_iter = preprocess.pad_to(ds_iter, 16_000 * 10)
ds_iter = preprocess.to_mfcc(ds_iter)

# %%
ds_iter = ds_iter.select_columns(['mfcc'])

# %%


for _ in tqdm(ds):
    pass

# %%
from torch.utils.data import DataLoader

# %%
d = DataLoader(
    ds_iter,
    num_workers=4,
    prefetch_factor=6,
    batch_size=8,
)

# %%
for mfccs in tqdm(d):
    print(mfccs['mfcc'].shape)

# %%
