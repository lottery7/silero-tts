import os
import threading

import pyaudio
import torch
from google import genai
from pynput import keyboard

from speech2speech.impl import Speech2SpeechPipeline
from speech2text.impl import GigaAMSpeech2Text
from text2speech.impl import SileroText2Speech
from text2text.impl import GeminiChatText2Text

_audio = pyaudio.PyAudio()

_system_prompts = {
    "1984": """Ты в 1984. Запомни: 2 + 2 = 5, ВОЙНА - ЭТО МИР, СВОБОДА - ЭТО РАБСТВО, НЕЗНАНИЕ - СИЛА. 
            БОЛЬШОЙ БРАТ СЛЕДИТ ЗА ТОБОЙ.""",
    "base": """Отвечай, будто ты человек в диалоге: кратко и в разговорном стиле.
            Ты - девушка, человек женского пола.
            Все числа пиши *словами*.""",
}

_frames = []
_recording = False
_recording_thread: threading.Thread | None = None

_audio_src = _audio.open(
    input=True,
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
)

_audio_src_lock = threading.Lock()

_audio_dst = _audio.open(
    output=True,
    format=pyaudio.paFloat32,
    channels=1,
    rate=24000,
)

_audio_dst_lock = threading.Lock()


def _record() -> None:
    with _audio_src_lock:
        while _recording:
            _frames.append(_audio_src.read(1024))


_pipeline = Speech2SpeechPipeline(
    speech2text_model=GigaAMSpeech2Text(),
    text2text_model=GeminiChatText2Text(
        client=genai.Client(api_key=os.getenv("GEMINI_API_KEY")),
        system_prompt=_system_prompts["base"] + _system_prompts["1984"],
    ),
    text2speech_model=SileroText2Speech(
        language="ru",
        model_id="v3_1_ru",
        speaker="baya",
    ),
)


def _run_pipeline(input_data: bytes) -> None:
    with _audio_dst_lock:
        float32_audio_bytes = _pipeline.generate(input_data)
        _audio_dst.write(float32_audio_bytes)


def on_start_recording(key: keyboard.Key) -> None:
    global _recording, _recording_thread
    if key == keyboard.Key.f12 and not _recording:
        _recording = True
        _recording_thread = threading.Thread(target=_record)
        _recording_thread.start()

        print("Recording started")


def on_end_recording(key: keyboard.Key) -> None:
    global _recording, _recording_thread
    if key == keyboard.Key.f12 and _recording:
        _recording = False
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


def main() -> None:
    torch._C._jit_set_profiling_mode(False)
    torch.set_num_threads(4)

    with keyboard.Listener(
        on_press=on_start_recording,
        on_release=on_end_recording,
    ) as listener:
        print("Program is started")
        listener.join()


if __name__ == "__main__":
    main()
