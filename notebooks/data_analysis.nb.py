# %% [markdown]
# # Data analysis

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torch
from IPython.display import Audio, display
from youtube_asr.utils import load_audio, audio_path

# %%
pd.set_option("display.max_rows", 500)
plt.rc("figure", figsize=(12, 8))

# %% [markdown]
# ## Getting to know the task
#
# - What do multi-labels mean? Are they order-invariant?
# - Listen to few examples

# %%
df = pd.read_csv("../data/dataset.tsv", sep="\t")

# %%
df.shape

# %%
df["labels"].unique()

# %%
df["labels"] = df["labels"].str.split(",").map(tuple)

# %%
audio_files = len(df["audio"].unique())
print(audio_files)


# %% [markdown]
# Notes:
# - not a cleanly annotated dataset
#     - labels don't include everything that happens
#         - e.g.
#             - `133` missing intrument,
#             - `681` missing wind,
#             - `78` missing speech,
#             - `124` missing vehicle, music,
#             - `753` and `931` missing speech
#             - `328` missing speech
#     - labels sometimes stand for events that take up minority of the recording
#         - e.g.
#             - `643` 1s taken up by tru label music,intrument, 9s unlabelled speech
#             - `931` 8s of unlabelled talking, 2s of labelled tool
#         - its not like events happen in the middle of the recording -- sometimes they are at the beginning, sometimes at the end
#     - few mislabellings (extras and errors)
#         - e.g.
#             - `191` extra animal
#             - `623` extra animal
#             - `231` baby noise is sometimes interpretted as animal
#             - `1523` extra vehicle
#             - `1899` rap being recognized as signing
# - speech includes multiple languages
# - multi-labels are order-invariant
#     - sometimes its impossible to say which comes first
#     - sometimes they happen at the same time
#     - if they are in order, the order doesn't need to correspond to the label order
#     - for some combination more than one order exists in the dataset
#         - e.g.
#             - `'Music,Musical instrument'` and `'Music,Musical instrument'`
#             - `'Speech', 'Animal'` and `'Animal', 'Speech'`
# - singing sometimes means just concert noise
# - there is a 'silence' label
# - not all clips are 10s
#     - e.g.
#         - `706` 8s
#         - `930` 9s
# - volume of clips is not the same
#     - e.g.
#         - `954` a lot quieter than `1304`, (same labels)
# - duplicates
#     - e.g.
#         - `3290` ('Animal', 'Speech', 'Music') and `139` ('Speech', 'Music')
#         - `7020` ('Music', 'Musical instrument', 'Speech') and (I know I heard it already, I just couldn't find the previous file)
#         - `3` ('Water',) and `111` ('Water',)
# - distorted audio
#     - e.g.
#         - `9753`

# %%
for labelset in df["labels"].unique():
    filenames = df[df["labels"] == labelset]["audio"]
    print(labelset, len(filenames))
    for index, filename in filenames.iloc[:5].items():
        print(f"\t{index}")
        display(Audio(audio_path(filename)))


# %% [markdown]
# ## Label distributions and train/dev split
#
# 1. deduplication -- merging duplicate recordings
# 2. Seeing the label distribution (both combination and pure)
# 3. splitting by label (not label combination)
#     1. explode by label, but keep mapping to original
#     2. stratified 80/20 split
#     3. merge by mapping
#     4. if sample is both in dev and train, keep that in dev w/ all labels from train as well
# 4. Check distributions and counts

# %% [markdown]
# ### Deduplication (try)
#
# I tried to deduplicate audio clips to clean-up the dataset. However, I haven't been able to find a method that works 100%.
#
# This is one, that worked the best:
# - creating MFCCs with large windows
# - splitting them, and
# - finding duplicates using kNN

# %% [markdown]
# #### How similar are those duplicates?


# %%
def _shift(wav, shift):
    return np.concatenate(
        [
            np.zeros(shift) if shift > 0 else [],
            wav,
            np.zeros(-shift) if shift < 0 else [],
        ]
    )[(-shift if shift < 0 else 0) : len(wav) + (-shift if shift < 0 else 0)]


# %%
load_audio(df["audio"][3290])

# %%
xss = np.arange(16_000 * 10)
aud1, _ = load_audio(df["audio"][3290])
aud2, _ = load_audio(df["audio"][139])

fig, ax = plt.subplots()
ax.plot(xss, aud1, alpha=0.4)
ax.plot(xss, _shift(aud2, 2550), alpha=0.4)

# %%
fig.savefig("./imgs/duplicate_audio_example.png", dpi=300, bbox_inches="tight")

# %%
Audio(audio_path(df["audio"][3290]))

# %%
Audio(audio_path(df["audio"][139]))

# %%
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tqdm.auto import tqdm
from collections import defaultdict

# %%
mfcc = torchaudio.transforms.MFCC(
    sample_rate=16_000,
    n_mfcc=13,
    log_mels=True,
    melkwargs={
        "n_fft": 4 * 4096,
        "win_length": 4 * 4096,
        "hop_length": 256,
        "n_mels": 64,
        "center": False,
    },
)

extra_data = []
for filename in tqdm(df["audio"]):
    wav, sr = load_audio(filename)
    extra_data.append(
        {
            "sr": sr,
            "length": len(wav) / sr,
            "mfcc": mfcc(torch.tensor(wav)).numpy(),
        }
    )

df = pd.concat((df, pd.DataFrame(extra_data)), axis=1)

# %%
df.describe()

# %%
df["mfcc"][0].shape

# %%
chunk_size = 512
overlap = 128

chunks = []
idxs = []

for idx, row in tqdm(df.iterrows()):
    for chunk_begin in range(0, row["mfcc"].shape[1], overlap):
        chunk_begin = max(0, min(chunk_begin, row["mfcc"].shape[1] - chunk_size))
        chunk = row["mfcc"][:, chunk_begin : chunk_begin + chunk_size]
        filters, chunk_frames = chunk.shape
        if chunk_frames < chunk_size:
            chunk = np.concatenate(
                [
                    chunk,
                    np.zeros((filters, chunk_size - chunk_frames)),
                ],
                axis=-1,
            )
        assert chunk.shape == (
            filters,
            chunk_size,
        ), f"Chunk shape: {chunk.shape}, {chunk_begin}, {row['mfcc'].shape}"
        chunks.append(chunk.flatten())
        idxs.append(idx)
# %%
len(chunks)

# %%
neighbors = KNeighborsClassifier(n_neighbors=64, n_jobs=8)
neighbors.fit(chunks, idxs)
# %%
audio_to_chunk_idxs = defaultdict(list)
for chunk_idx, audio_idx in enumerate(idxs):
    audio_to_chunk_idxs[audio_idx].append(chunk_idx)

# %%
audio_to_chunk_idxs[3290]

# %%
query_audio_idx = 3290
dist, predicted_chunk_idxs = neighbors.kneighbors(
    [chunks[query_idx] for query_idx in audio_to_chunk_idxs[query_audio_idx]]
)

all_dists = dist.flatten()
dists_sort_idxs = np.argsort(all_dists)
predicted_chunk_idxs = predicted_chunk_idxs.flatten()

pred_audio = defaultdict(list)
for pred_idx in dists_sort_idxs:
    chunk_idx = predicted_chunk_idxs[pred_idx]
    dist = all_dists[pred_idx]
    audio_idx = idxs[chunk_idx]
    if audio_idx == query_audio_idx:
        continue
    pred_audio[audio_idx].append(dist)

means = np.array([np.mean(dists) for dists in pred_audio.values()])
means_sort_idxs = np.argsort(means)
pred_audio_idxs = np.array(list(pred_audio.keys()))

print(f"Query audio {query_audio_idx}: {df['labels'][query_audio_idx]}")
display(Audio(audio_path(df["audio"][query_audio_idx])))
print("-" * 40)
for pred_idx in means_sort_idxs:
    mean_dist = means[pred_idx]
    audio_idx = pred_audio_idxs[pred_idx]
    count = len(pred_audio[audio_idx])
    print(f"audio {audio_idx}: {mean_dist:.2f}, x{count}, {df['labels'][audio_idx]}")
    display(Audio(audio_path(df["audio"][audio_idx])))

# %% [markdown]
# ### Label distributions

# %%
from collections import Counter


# %%
def plot_label_distribution(df: pd.DataFrame, hue: str | None = None) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 17))
    hue_kwargs = {}
    if hue is not None:
        hue_kwargs = {
            "hue": df[hue],
            "common_norm": False,
            "element": "step",
        }

    sns.histplot(
        y=df["labels"].apply(lambda tup: ",".join(tup)),
        ax=ax1,
        stat="percent",
        **hue_kwargs,
    )
    ax1.set_title("Label combination")

    df_exploded = df.explode("labels")
    sns.histplot(df_exploded, y="labels", ax=ax2, stat="percent", **hue_kwargs)
    ax2.set_title("Labels")

    return fig


# %%

plot_label_distribution(df)

# %% [markdown]
# ### Splitting to train and dev

# %%
df_exploded = df.reset_index().rename(columns={"index": "audio_idx"}).explode("labels")

# %%
df_exploded.shape

# %%
from sklearn.model_selection import train_test_split

# %%
audio_idxs_train, audio_idxs_dev = train_test_split(
    df_exploded["audio_idx"],
    train_size=0.8,
    random_state=42,
    stratify=df_exploded["labels"],
)

# %%
len(audio_idxs_train), len(set(audio_idxs_train))

# %%
len(audio_idxs_dev), len(set(audio_idxs_dev))

# %%
len(set(audio_idxs_train) & set(audio_idxs_dev))

# %%
audio_idxs_train = list(set(audio_idxs_train) - set(audio_idxs_dev))
audio_idxs_dev = list(set(audio_idxs_dev))

# %%
len(audio_idxs_train), len(audio_idxs_dev), df.shape[0]

# %%
train_df = df.loc[audio_idxs_train][["audio", "labels"]]
dev_df = df.loc[audio_idxs_dev][["audio", "labels"]]

train_df.shape, dev_df.shape


# %%
def save_df(df: pd.DataFrame, path: str) -> None:
    df["labels"] = df["labels"].apply(lambda tup: ",".join(tup))
    df["audio_idx"] = df.index
    df.to_csv(path, sep="\t", index=False)


# %%
save_df(train_df, "../data/dataset_train.tsv")

# %%
save_df(dev_df, "../data/dataset_dev.tsv")

# %%
df.loc[audio_idxs_train, "split"] = "train"
df.loc[audio_idxs_dev, "split"] = "dev"

# %%
fig = plot_label_distribution(df, hue="split")

# %%
fig.savefig("./imgs/dev_train_distributions.png", dpi=300, bbox_inches="tight")
