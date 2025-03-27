import os
import threading

import numpy as np
import pyaudio
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

audio_source_lock = threading.Lock()

audio_target = pa.open(
    output=True,
    format=pyaudio.paFloat32,
    channels=1,
    rate=24000,
)

audio_target_lock = threading.Lock()


def _read_start_sound() -> bytes:
    audio_path = os.path.join("audio", "start_sound.wav")
    _, wav_array = wavfile.read(audio_path)
    wav_array = wav_array.astype(np.float32) / 32768
    return wav_array.tobytes()


start_recording_sound_bytes = _read_start_sound()
