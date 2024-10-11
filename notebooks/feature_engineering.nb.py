# %% [markdown]
# # Analysis of train data
#
# Explore training data and its features.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

from youtube_asr.dataset import load_dataset, load_pandas, SR
from youtube_asr.utils import load_audio
from youtube_asr import preprocess
import torch
import matplotlib.pyplot as plt
import librosa
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display, Audio
import ipywidgets as widgets
import torchaudio

from typing import Any

# %%
plt.rc("figure", figsize=(12, 6))

# %% [markdown]
# ## Features
#
# %%
df = load_pandas("train")

# %% [markdown]
# ### Distribution of loudness per label
# %%


def _gen_loudness(filename: str) -> float:
    wav, sr = load_audio(filename, np=False)
    return torchaudio.functional.loudness(wav, sr).item()


df["loudness"] = [_gen_loudness(filename) for filename in tqdm(df["audio"])]

# %%
df_tmp = (
    df[["loudness", "labels"]].explode("labels").reset_index().drop(columns="index")
)

# %%
df_tmp

# %%
ax = sns.boxplot(df_tmp, x="loudness", y="labels", hue="labels")
# %%
ax.figure.savefig("./imgs/loud_dist.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ### RMS

# %%
import librosa


# %%
def _gen_rms(filename: str) -> float:
    wav, _ = load_audio(filename)
    return librosa.feature.rms(y=wav, frame_length=1024, hop_length=1024).mean()


df["rms"] = [_gen_rms(filename) for filename in tqdm(df["audio"])]

# %%

df_tmp = df.explode("labels").reset_index().drop(columns="index")

# %%
sns.boxplot(df_tmp, x="rms", y="labels", hue="labels")


# %% [markdown]
# ### ZCR


# %%
def _gen_zcr(filename: str) -> float:
    wav, _ = load_audio(filename)
    return librosa.feature.zero_crossing_rate(
        y=wav, frame_length=1024, hop_length=1024
    ).mean()


df["zcr"] = [_gen_zcr(filename) for filename in tqdm(df["audio"])]

# %%

df_tmp = df.explode("labels").reset_index().drop(columns="index")

# %%
ax = sns.boxplot(df_tmp, x="zcr", y="labels", hue="labels")

# %%
ax.figure.savefig('./imgs/zcr_dist.png', dpi=300, bbox_inches='tight')

# %% [markdown]
# ### Max and min frequences
# Maybe some frequencies can be omitted?
# %%
spectogram_transform = torchaudio.transforms.Spectrogram(
    n_fft=1024, win_length=1024, hop_length=256, power=2
)
spec = spectogram_transform(load_audio(df["audio"], np=False)[0])

# %%
spec.shape

# %%
max_freq = torch.max(spec[0], dim=1)

# %%
max_freq.values.shape

# %%
plt.plot(max_freq.values)
plt.plot((max_freq.values < 50) * 10000 + 0.01)
plt.yscale("log")

# %%
librosa.display.specshow(spec[0].numpy(), sr=16_000, hop_length=256, y_axis="hz")

# %%
# Computing min and maximum frequency band with amplitutdes above trheshold
threshold = 50

min_frequency_above = 0
max_frequency_above = 0
all_below_thres = 0
for _, row in tqdm(df.iterrows()):
    spec = spectogram_transform(load_audio(row["audio"], np=False))
    maxs, _ = spec[0].max(dim=1)
    above_thres = torch.nonzero(maxs > threshold).squeeze()
    if above_thres.ndim > 0 and len(above_thres) > 0:
        min_frequency_above = min(min_frequency_above, above_thres[0])
        max_frequency_above = max(max_frequency_above, above_thres[-1])
    else:
        all_below_thres += 1
# %%
all_below_thres

# %%
min_frequency_above * 16_000 / 1024

# %%
max_frequency_above * 16_000 / 1024

# %%

# %% [markdown]
# ### Visualization of MFCCs
# Is there some structure in MFCCs?
# %%

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SR,
    n_mfcc=24,
    log_mels=True,
    melkwargs=dict(
        n_fft=4 * 4096,
        win_length=4 * 4096,
        hop_length=512,
        n_mels=64,
    ),
)

# %%
from sklearn.decomposition import PCA

# %%
mfccs = [
    mfcc_transform(load_audio(filename, np=False)[0]).numpy()[0]
    for filename in tqdm(df["audio"])
]

# %%
longest = max(mfcc.shape[1] for mfcc in mfccs)

# %%
longest


# %%
def _pad_mfcc(mfcc: np.ndarray, length: int) -> np.ndarray:
    filters, mfcc_len = mfcc.shape
    if mfcc_len == length:
        return mfcc

    return np.concatenate(
        [
            mfcc,
            np.zeros((filters, length - mfcc_len)),
        ],
        axis=-1,
    )


mfccs = np.stack([_pad_mfcc(mfcc, longest).reshape((-1,)) for mfcc in mfccs])

# %%
mfccs.shape

# %%
pca = PCA(n_components=0.9, svd_solver="full")
mfccs_reduced = pca.fit_transform(mfccs)

# %%
len(pca.components_)

# %%
plt.plot(pca.explained_variance_ratio_)

# %%
from sklearn.manifold import TSNE

# %%
tsne = TSNE()
mfccs_2d = tsne.fit_transform(mfccs_reduced)

# %%
mfccs_2d.shape

# %%
df_tmp = pd.DataFrame(
    {
        "x": mfccs_2d[:, 0],
        "y": mfccs_2d[:, 1],
        "labels": df["labels"],
    }
)
df_tmp = df_tmp.explode("labels").reset_index().drop(columns="index")

# %%
ax = sns.scatterplot(df_tmp, x="x", y="y", hue="labels")
# %%
ax.figure.savefig("./imgs/pca_mfccs.png")
# %% [markdown]
# ### Looking at MFCCs
# %%
import ipywidgets as widgets

# %%
mel_kwargs = dict(n_fft=4096, win_length=1024, hop_length=512, n_mels=128)
mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_kwargs)


mfcc_transform = torchaudio.transforms.MFCC(
    n_mfcc=96, log_mels=True, melkwargs=mel_kwargs
)

rows_iter = iter(df.iterrows())
# %%
bttn = widgets.Button(description="Next")
wav_out = widgets.Output()
mel_out = widgets.Output()
mfcc_out = widgets.Output()

row = None


def display_next(bttn: widgets.Button):
    global row
    mfcc_out.clear_output()
    wav_out.clear_output()
    mel_out.clear_output()

    with wav_out:
        idx, row = next(rows_iter)
        wav = load_audio(row["audio"], np=False)[0]

        fig, ax = plt.subplots(figsize=(20, 1.5))
        ax.plot(wav[0].numpy())
        ax.set_title(f"{idx}: {row['labels']}")
        display(fig)
    with mel_out:
        fig, ax = plt.subplots(figsize=(20, 3))
        mel_spec = mel_spec_transform(wav)
        ax.imshow(mel_spec[0].numpy())
        display(fig)
    with mfcc_out:
        fig, ax = plt.subplots(figsize=(20, 3))
        mfcc = mfcc_transform(wav)
        ax.imshow(mfcc[0].numpy())
        display(fig)


bttn.on_click(display_next)
display(widgets.VBox([bttn, wav_out, mel_out, mfcc_out]))

# %% [markdown]
# ## View transformations
#
# View what do the transformations do with the input.
# %% [markdown]
# ### Time wrap


# %%

rows_iter = iter(df.iterrows())

out = widgets.Output()

next_bttn = widgets.Button(description="Next")


def _display_next(bttn):
    out.clear_output()
    with out:
        idx, row = next(rows_iter)
        wav, _ = load_audio(row["audio"], np=False)
        time_wrapped = preprocess.time_wrap(wav)
        print(f"Clean")
        display(Audio(wav.squeeze(0), rate=16_000))
        print(f"Time wrapped")
        display(Audio(time_wrapped.squeeze(0), rate=16_000))


next_bttn.on_click(_display_next)

widgets.VBox([next_bttn, out])

# %% [markdown]
# ### Upsampling

# %%
from youtube_asr import preprocess


rows_iter = iter(df.iterrows())

out = widgets.Output()

next_bttn = widgets.Button(description="Next")

upsample_factors = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
all_labels = sorted(
    {label for label_comb in df["labels"].unique() for label in label_comb}
)


def _display_next(bttn):
    out.clear_output()
    with out:
        idx, row = next(rows_iter)
        targets = torch.tensor([[int(label in row["labels"]) for label in all_labels]])
        repeats = preprocess.upsample_to_repeats(targets, upsample_factors)
        print(
            f"Labels: {row['labels']}, factors: {upsample_factors[targets[0].to(bool)]}, repeat: {repeats}"
        )


next_bttn.on_click(_display_next)

widgets.VBox([next_bttn, out])
# %% [markdown]
# ### Frequency & Time masking
# %%

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SR,
    n_mfcc=24,
    log_mels=True,
    melkwargs=dict(
        n_fft=4 * 4096,
        win_length=4 * 4096,
        hop_length=512,
        n_mels=64,
    ),
)

rows_iter = iter(df.iterrows())

out = widgets.Output()

next_bttn = widgets.Button(description="Next")

freq_max_range = 32
time_max_range = 120


def _display_next(bttn):
    out.clear_output()
    with out:
        idx, row = next(rows_iter)
        wav = load_audio(row["audio"], np=False)[0]
        mfcc = mfcc_transform(wav)
        print(mfcc.shape)
        freq_masked = preprocess.random_dimension_mask(mfcc, freq_max_range, 1)
        time_masked = preprocess.random_dimension_mask(mfcc, time_max_range, 2)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(22, 8))

        ax1.set_title(f"Audio idx: {idx}, labels: {row['labels']}")
        ax1_mappable = ax1.imshow(mfcc[0])
        plt.colorbar(ax1_mappable, ax=ax1)

        ax2.set_title("Frequency masked")
        ax2_mappable = ax2.imshow(freq_masked[0])
        plt.colorbar(ax2_mappable, ax=ax2)

        ax3.set_title("Time masked")
        ax3_mappable = ax3.imshow(time_masked[0])
        plt.colorbar(ax3_mappable, ax=ax3)
        display(fig)


next_bttn.on_click(_display_next)

widgets.VBox([next_bttn, out])

# %% [markdown]
# ### Normalization
# %%
rows_iter = iter(df.iterrows())

out = widgets.Output()

next_bttn = widgets.Button(description="Next")


def _display_next(bttn):
    out.clear_output()
    with out:
        idx, row = next(rows_iter)
        wav = load_audio(row["audio"], np=False)[0]
        mfcc = mfcc_transform(wav)
        normed = preprocess.norm_non_batch_dims(mfcc.unsqueeze(0))[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 8))
        ax1.set_title(f"Audio idx: {idx}, labels: {row['labels']}")
        ax1_mappable = ax1.imshow(mfcc[0])
        plt.colorbar(ax1_mappable, ax=ax1)

        ax2.set_title("Normed")
        ax2_mappable = ax2.imshow(normed[0])
        plt.colorbar(ax2_mappable, ax=ax2)
        display(fig)


next_bttn.on_click(_display_next)

widgets.VBox([next_bttn, out])
# %% [markdown]
# ## One-off computations

# %% [markdown]
# ### Computing upsample factors

# %%
from youtube_asr.dataset import load_pandas

# %%
df_train = load_pandas("train")

# %%
df_train["labels"] = df_train["labels"].apply(lambda labels: tuple(labels.split(",")))

# %%
df_exploded = df_train.explode("labels")

# %%
from collections import Counter

# %%
label_counter = Counter(df_exploded["labels"])

# %%
label_counter

# %%
total = len(df_exploded)
total

# %%
max_count = max(label_counter.values())
max_count

# %%
to_equal_upsample_factors = {k: max_count / count for k, count in label_counter.items()}
to_equal_upsample_factors

# %%
equal_counts = {
    k: int(count * to_equal_upsample_factors[k]) for k, count in label_counter.items()
}
equal_counts

# %%
[to_equal_upsample_factors[k] for k in sorted(label_counter)]

# %%
sorted(label_counter)
