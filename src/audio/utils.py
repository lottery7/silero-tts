import os
import tempfile
import wave

import numpy as np
import silero_vad
import torch
import torchaudio.transforms as T
from scipy.io import wavfile

from config import CONFIG
from utils import torch_single_thread

from .constants import audio_target, audio_target_lock

__all__ = ["create_wav_from_bytes"]


def create_wav_from_bytes(
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
    silence = np.zeros(
        round(CONFIG.audio.output.sample_rate * duration),
        dtype=np.float32,
    )
    with audio_target_lock:
        audio_target.write(silence.tobytes())


def calc_audio_length(
    num_bytes: int,
    channels: int,
    sample_width: int,
    sample_rate: int,
) -> float:
    return num_bytes / (channels * sample_width * sample_rate)


_silero_vad = silero_vad.load_silero_vad()


@torch_single_thread
def split_vad(
    audio_path: str,
    max_speech_duration_s: float = float("inf"),
) -> list[tuple[bytes, int]]:
    # Load audio
    wav = silero_vad.read_audio(audio_path, sampling_rate=16000)

    # Split on segments
    speech_timestamps = silero_vad.get_speech_timestamps(
        wav,
        _silero_vad,
        sampling_rate=16000,
        max_speech_duration_s=max_speech_duration_s,
        return_seconds=False,
    )
    print(speech_timestamps)

    # Aggregate results
    result: list[tuple[bytes, int]] = []
    for ts in speech_timestamps:
        _wav = wav[ts["start"] : ts["end"]]
        _wav = (_wav * 32767).numpy().astype(np.int16)
        result.append((_wav.tobytes(), 16000))

    return result


def resample(audio_input: np.ndarray, base_sr: int, target_sr: int) -> np.ndarray:
    if base_sr == target_sr:
        return audio_input

    # Transform to Tensor
    waveform = torch.from_numpy(audio_input)

    # Resample
    resampler = T.Resample(
        orig_freq=base_sr,
        new_freq=target_sr,
    )
    waveform: torch.Tensor = resampler(waveform)

    # Transform back to numpy
    waveform = waveform.numpy()

    return waveform


def _read_start_sound() -> np.ndarray:
    # Read bytes
    audio_path = os.path.join("audio", "start_sound.wav")
    sample_rate, waveform = wavfile.read(audio_path)

    assert waveform.dtype == np.int16

    # Resample if needed
    waveform = resample(waveform, sample_rate, CONFIG.audio.input.sample_rate)

    return waveform


_start_recording_sound_bytes = _read_start_sound().tobytes()


def play_start_sound() -> None:
    with audio_target_lock:
        audio_target.write(_start_recording_sound_bytes)
