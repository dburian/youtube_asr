from __future__ import annotations
import random
from typing import TYPE_CHECKING

import torchaudio
from copy import deepcopy
import torch
import datasets as hf_datasets

if TYPE_CHECKING:
    from typing import Any, TypeAlias

    Dataset: TypeAlias = hf_datasets.Dataset | hf_datasets.IterableDataset


SR = 16_000


def _transform_map(
    sample: dict[str, Any], *, transform: torch.nn.Module, key: str
) -> dict[str, Any]:
    return {key: transform(sample["wav"])}


def to_spectogram(
    ds: Dataset,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    power: int = 2,
) -> Dataset:
    transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=power
    )

    return ds.map(
        _transform_map,
        fn_kwargs={"transform": transform, "key": "spectogram"},
    )


def to_mel_spectogram(
    ds: Dataset,
    sample_rate: int = SR,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
) -> Dataset:
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    return ds.map(
        _transform_map,
        fn_kwargs={
            "transform": transform,
            "key": "mel_spectogram",
        },
    )


def to_mfcc(
    ds: Dataset,
    n_mfcc: int = 64,
    log_mels: bool = True,
    sample_rate: int = SR,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
) -> Dataset:
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        log_mels=log_mels,
        melkwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
        },
    )

    return ds.map(
        _transform_map,
        fn_kwargs={
            "transform": transform,
            "key": "mfcc",
        },
    )


def _pad_map(sample: dict[str, Any], target_length: int) -> dict[str, Any]:
    wav = sample["wav"]
    channels, length = wav.shape
    if length < target_length:
        wav = torch.concat(
            [wav.clone().detach(), torch.zeros((channels, target_length - length))],
            dim=-1,
        )
    elif length > target_length:
        wav = wav[:target_length]

    return {"wav": wav}


def pad_to(
    ds: Dataset,
    samples: int,
) -> Dataset:
    return ds.map(
        _pad_map,
        fn_kwargs={"target_length": samples},
    )


def time_wrap(wav: torch.Tensor) -> torch.Tensor:
    """Randomly rotates signal along the time dimension."""
    length = wav.shape[1]
    wrap_idx = torch.randint(0, length, (1,))
    return torch.concat(
        [wav[:, wrap_idx:], wav[:, :wrap_idx]],
        dim=-1,
    )


def upsample_to_repeats(
    targets: torch.Tensor, upsample_factors: torch.Tensor
) -> torch.Tensor:
    """Computes random number of repeats based on upsample factors for targets."""
    max_upsamples = (targets * upsample_factors.unsqueeze(0)).sum(dim=1) / targets.sum(
        dim=1
    )

    return torch.floor(
        torch.rand((max_upsamples.shape[0],)) * (max_upsamples - 1) + 1
    ).to(torch.int32)


def random_dimension_mask(
    input: torch.Tensor, max_range: int, dim: int
) -> torch.Tensor:
    """Random masks out a strip along `dim` of `input`."""
    length_in_mask_dim = input.shape[dim]
    masked_range = random.randint(0, max_range)
    begin_mask = random.randint(0, length_in_mask_dim - masked_range)

    mask = torch.ones_like(input)
    if dim == 1:
        mask[:, begin_mask : begin_mask + masked_range] = 0
    elif dim == 2:
        mask[:, :, begin_mask : begin_mask + masked_range] = 0

    return input * mask


def norm_non_batch_dims(input: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, input.ndim))
    means = input.mean(dim=dims, keepdim=True)
    stds = input.std(dim=dims, keepdim=True)

    return (input - means) / stds


class Preprocess:
    """Main class for signal preprocessing for training purposes."""

    def __init__(
        self,
        target_names: list[str],
        pad_to: int,
        mfcc_kwargs: dict[str, Any],
        time_wrap_prob: float = 0.0,
        mask_frequency_prob: float = 0.0,
        max_frequency_mask_range: int = 5,
        mask_time_prob: float = 0.0,
        max_time_mask_range: int = 80,
        upsample_factors: dict[str, float] | None = None,
        norm_mfcc: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        target_names: list[str]
            List of label names for which to generate targets.
        pad_to: int
            Number of samples to which pad each waveform.
        mfcc_kwargs: dict[str, Any]
            Key word arguments passed to MFCC transformation. Refer to
            `torchaudio.transforms.MFCC`.
        time_wrap: float
            Probablity to apply `time_wrap` to a sample.
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
        """

        mfcc_kwargs = deepcopy(mfcc_kwargs)
        mel_kwargs = {
            "n_fft": mfcc_kwargs.pop("n_fft", 1024),
            "win_length": mfcc_kwargs.pop("win_length", 1024),
            "hop_length": mfcc_kwargs.pop("hop_length", 1024),
            "n_mels": mfcc_kwargs.pop("n_mels", 64),
        }
        mfcc_kwargs["melkwargs"] = mel_kwargs
        self.mfcc_transform = torchaudio.transforms.MFCC(**mfcc_kwargs)

        self.time_wrap_prob = time_wrap_prob
        self.mask_frequency_prob = mask_frequency_prob
        self.max_frequency_mask_range = max_frequency_mask_range
        self.mask_time_prob = mask_time_prob
        self.max_time_mask_range = max_time_mask_range
        self.upsample_factors = (
            torch.tensor([upsample_factors[k] for k in target_names])
            if upsample_factors is not None
            else None
        )
        self.norm_mfcc = norm_mfcc
        self.target_names = target_names
        self.pad_to = pad_to

    def _upsample(
        self, targets: torch.Tensor, wavs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.upsample_factors is None:
            return targets, wavs

        batch_repeats = upsample_to_repeats(targets, self.upsample_factors)

        return (
            torch.repeat_interleave(targets, batch_repeats, dim=0),
            [
                v.clone()
                for v, repeats in zip(wavs, batch_repeats)
                for _ in range(repeats)
            ],
        )

    def _time_wrap(self, wavs: list[torch.Tensor]) -> list[torch.Tensor]:
        wrapped_wavs = []
        for wav in wavs:
            if random.random() > self.time_wrap_prob:
                wrapped_wavs.append(wav)
                continue

            wrapped_wavs.append(time_wrap(wav))

        return wrapped_wavs

    def _pad_to(self, wavs: list[torch.Tensor]) -> torch.Tensor:
        padded_wavs = []
        for wav in wavs:
            channels, length = wav.shape
            if length < self.pad_to:
                wav = torch.concat(
                    [wav, torch.zeros((channels, self.pad_to - length))],
                    dim=-1,
                )
            elif length > self.pad_to:
                wav = wav[:, : self.pad_to]
            padded_wavs.append(wav)

        return torch.stack(padded_wavs, dim=0)

    def _mask(
        self, input: torch.Tensor, prob: float, max_range: int, dim: int
    ) -> torch.Tensor:
        batch_size = input.shape[0]
        for i in range(batch_size):
            if random.random() > prob:
                continue

            input[i] = random_dimension_mask(input[i], max_range, dim)

        return input

    def _process_one_batch(self, sample_batch: dict[str, Any]) -> dict[str, Any]:
        targets = torch.tensor(
            [
                [sample_labels[label] for label in self.target_names]
                for sample_labels in sample_batch["labels"]
            ],
            dtype=torch.float32,
        )
        wavs = [torch.tensor(wav) for wav in sample_batch["wav"]]

        targets, wavs = self._upsample(targets=targets, wavs=wavs)

        if self.time_wrap_prob > 0:
            wavs = self._time_wrap(wavs)

        wavs_collated = self._pad_to(wavs)
        mfccs = self.mfcc_transform(wavs_collated)

        if self.norm_mfcc:
            mfccs = norm_non_batch_dims(mfccs)

        if self.mask_frequency_prob > 0:
            mfccs = self._mask(
                input=mfccs,
                prob=self.mask_frequency_prob,
                max_range=self.max_frequency_mask_range,
                dim=1,
            )

        if self.mask_time_prob > 0:
            mfccs = self._mask(
                input=mfccs,
                prob=self.mask_time_prob,
                max_range=self.max_time_mask_range,
                dim=2,
            )

        return {"mfcc": mfccs, "targets": targets}

    def process(self, ds: Dataset) -> Dataset:
        """Applies defined preprocessing to a dataset."""

        return ds.map(
            self._process_one_batch,
            remove_columns=["audio_idx", "labels", "wav", "audio"],
            batched=True,
            batch_size=32,
            features=hf_datasets.Features(
                {
                    "targets": hf_datasets.Sequence(
                        hf_datasets.Value("int32"), length=len(self.target_names)
                    ),
                    "mfcc": hf_datasets.Sequence(
                        hf_datasets.Sequence(
                            hf_datasets.Sequence(hf_datasets.Value("float32"))
                        ),
                        length=1,
                    ),
                }
            ),
        )
