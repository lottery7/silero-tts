import threading

import pyaudio

from silero_tts.config import CONFIG

__all__ = [
    "pa",
    "audio_source",
    "audio_source_lock",
    "audio_target",
    "audio_target_lock",
]

pa = pyaudio.PyAudio()

audio_source = pa.open(
    input=True,
    format=pyaudio.paInt16,
    channels=CONFIG.audio.input.channels,
    rate=CONFIG.audio.input.sample_rate,
)

audio_source_lock = threading.RLock()

audio_target = pa.open(
    output=True,
    format=pyaudio.paInt16,
    channels=CONFIG.audio.input.channels,
    rate=CONFIG.audio.input.sample_rate,
)

audio_target_lock = threading.RLock()
