import tempfile
import wave

import numpy as np

from globals.audio import audio_target, audio_target_lock

__all__ = ["create_wf_from_bytes"]


def create_wf_from_bytes(
    input_data: bytes,
    channels: int,
    sample_width: int,
    sample_rate: int,
) -> str:

    # Create temporary .wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_filename = temp_wav.name

    # Write input_data to temporary file
    with wave.open(temp_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(input_data)

    return temp_filename


def play_silence(duration: float) -> None:
    sample_rate = 24000
    silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
    with audio_target_lock:
        audio_target.write(silence.tobytes())
