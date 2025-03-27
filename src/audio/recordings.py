import threading
from queue import Queue

from audio.utils import play_silence
from globals.ai import s2s_pipeline
from globals.audio import *

__all__ = ["try_start_recording", "try_end_recording"]

_frames = []
_is_recording = False
_recording_thread: threading.Thread | None = None
_audio_play_queue = Queue()


def _record_from_mic() -> None:
    with audio_source_lock:
        while _is_recording:
            _frames.append(audio_source.read(1024))


def _play_speech_from_queue() -> None:
    while True:
        data: bytes = _audio_play_queue.get()
        with audio_target_lock:
            audio_target.write(data)
            play_silence(0.1)


threading.Thread(target=_play_speech_from_queue, daemon=True).start()


def _run_pipeline(input_data: bytes) -> None:
    for float32_audio_bytes in s2s_pipeline.generate(input_data):
        _audio_play_queue.put_nowait(float32_audio_bytes)


def try_start_recording() -> None:
    global _is_recording, _recording_thread
    if not _is_recording:
        _is_recording = True
        _recording_thread = threading.Thread(target=_record_from_mic)
        _recording_thread.start()
        print("Recording started")

        with audio_target_lock:
            audio_target.write(start_recording_sound_bytes)


def try_end_recording() -> None:
    global _is_recording, _recording_thread
    if _is_recording:
        _is_recording = False
        _recording_thread.join()
        print("Recording ended")

        if _frames:
            recorded_data = b"".join(_frames)
            _frames.clear()
            threading.Thread(
                daemon=True,
                target=_run_pipeline,
                args=[recorded_data],
            ).start()
