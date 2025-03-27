import os
import threading

import numpy as np
import pyaudio
import torch
import torchaudio.transforms as T
from scipy.io import wavfile

__all__ = [
    "pa",
    "audio_source",
    "audio_source_lock",
    "audio_target",
    "audio_target_lock",
    "start_recording_sound_bytes",
]

pa = pyaudio.PyAudio()

audio_source = pa.open(
    input=True,
    format=pyaudio.paInt16,
    channels=1,
    rate=24000,
)

audio_source_lock = threading.RLock()

audio_target = pa.open(
    output=True,
    format=pyaudio.paFloat32,
    channels=1,
    rate=24000,
)

audio_target_lock = threading.RLock()


def _read_start_sound() -> bytes:
    # Read bytes
    audio_path = os.path.join("audio", "start_sound.wav")
    sample_rate, waveform = wavfile.read(audio_path)

    # Transform to float32
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32) / 32768

    # Resample if needed
    if sample_rate != 24000:
        # Transform to Tensor
        waveform = torch.from_numpy(waveform)

        # Resample
        resampler = T.Resample(orig_freq=sample_rate, new_freq=24000)
        waveform: torch.Tensor = resampler(waveform)

        # Transform back to numpy
        waveform = waveform.numpy()

    return waveform.tobytes()


start_recording_sound_bytes = _read_start_sound()
