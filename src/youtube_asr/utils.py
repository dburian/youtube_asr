from __future__ import annotations

import os
from typing import TYPE_CHECKING
import torchaudio

from youtube_asr.dataset import DATA_PATH

if TYPE_CHECKING:
    import torch
    import numpy as np


def audio_path(filename: str) -> str:
    return os.path.join(DATA_PATH, "audio", f"{filename}.wav")


def load_audio(filename: str, np: bool = True) -> tuple[torch.Tensor | np.ndarray, int]:
    signal, sr = torchaudio.load(audio_path(filename))
    assert signal.shape[0] == 1, f"Signal '{filename}' has more than one channel."
    if np:
        signal = signal.squeeze(0).numpy()

    return signal, sr
