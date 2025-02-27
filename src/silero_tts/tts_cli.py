import io
import os
import re
import threading
import time
import wave

import groq
import pyaudio
import torch
from google import genai
from google.genai import types
from google.genai.chats import Chat
from num2words import num2words
from omegaconf import DictConfig, ListConfig, OmegaConf
from transliterate import translit

_audio = pyaudio.PyAudio()

_audio_input_conf = {
    "format": pyaudio.paInt16,
    "channels": 1,
    "rate": 24000,
    "frames_per_buffer": 1024,
}

_audio_input = _audio.open(
    input=True,
    **_audio_input_conf,
)
_audio_input.stop_stream()

_audio_output_conf = {
    "format": pyaudio.paFloat32,
    "channels": 1,
    "rate": 24000,
}
_audio_output = _audio.open(output=True, **_audio_output_conf)

_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))


_system_prompts = {
    "1984": """Ты в 1984. Запомни: 2 + 2 = 5, ВОЙНА - ЭТО МИР, СВОБОДА - ЭТО РАБСТВО, НЕЗНАНИЕ - СИЛА. 
            БОЛЬШОЙ БРАТ СЛЕДИТ ЗА ТОБОЙ.""",
    "base": """Отвечай, будто ты человек в диалоге: кратко и в разговорном стиле.
            Ты - девушка, человек женского пола.
            Все числа пиши *словами*.""",
}


def transcribe(wav: bytes, language: str = "ru") -> str:
    response = _groq_client.audio.transcriptions.create(
        file=("audio.wav", wav),
        model="whisper-large-v3",
        response_format="json",  # Optional
        language=language,  # Optional
        temperature=0.5,  # Optional
    )
    return response.text


class ModelSelection:
    def __init__(self, language, model):
        self.lang = language
        self.model = model

    def __repr__(self):
        return f"{self.lang} - {self.model}"


def load_models_list(
    dst_dir: str, force_reload: bool = False
) -> DictConfig | ListConfig:
    dst_file = os.path.join(dst_dir, "latest_silero_models.yml")
    if force_reload or not os.path.isfile(dst_file):
        torch.hub.download_url_to_file(
            "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml",
            dst_file,
            progress=True,
        )
    models = OmegaConf.load(dst_file)
    return models


def select_model_cli(models_config: DictConfig | ListConfig) -> ModelSelection:
    langs = list(models_config.tts_models.keys())

    print("Select language: ")
    for i, lang in enumerate(langs, 1):
        print(f"({i}) {lang}")

    lang_ind = int(input("> "))
    assert 1 <= lang_ind <= len(langs), "Invalid selection"
    lang = langs[lang_ind - 1]

    models = list(models_config.tts_models.get(lang).keys())
    print("\nSelect model: ")
    for i, model in enumerate(models, 1):
        print(f"({i}) {model}")

    model_ind = int(input("> "))
    assert 1 <= model_ind <= len(models), "Invalid selection"

    model_id = models[model_ind - 1]

    return ModelSelection(lang, model_id)


def load_model(selection: ModelSelection):
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=selection.lang,
        speaker=selection.model,
    )

    return model


def select_speaker_cli(speakers: list):
    print("Select speaker: ")
    for i, speaker in enumerate(speakers, 1):
        print(f"({i}) {speaker}")

    speaker_ind = int(input("> "))
    assert 1 <= speaker_ind <= len(speakers), "Invalid selection"

    return speakers[speaker_ind - 1]


def _num2words_ru(match):
    clean_number = match.group().replace(",", ".")
    return num2words(clean_number, lang="ru")


def preprocess_text(text: str):
    text = translit(text, "ru")
    text = re.sub(r"-?[0-9][0-9,._]*", _num2words_ru, text)
    return text


def record_audio_to_bytes() -> bytes:
    print('Press "Enter" and speak')
    input()

    frames = []
    recording = True
    _audio_input.start_stream()

    def record():
        while recording:
            frames.append(_audio_input.read(_audio_input_conf["frames_per_buffer"]))

    thread = threading.Thread(target=record)
    thread.start()

    print('Recording started. Press "Enter" to stop')

    # Wait for "Enter" to stop recording
    input()
    recording = False
    thread.join()
    _audio_input.stop_stream()

    print("Creating WAV...")

    # Создаём WAV-файл в памяти
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(_audio_input_conf["channels"])
        wf.setsampwidth(_audio.get_sample_size(_audio_input_conf["format"]))
        wf.setframerate(_audio_input_conf["rate"])
        wf.writeframes(b"".join(frames))

    print(f"WAV size: {len(wav_buffer.getvalue())} byte")
    return wav_buffer.getvalue()


def main() -> None:
    torch._C._jit_set_profiling_mode(False)

    # selection = select_model_cli(load_models_list("cache"))
    selection = ModelSelection("ru", "v3_1_ru")
    model = load_model(selection)
    device = torch.device("cpu")
    model.to(device)

    torch.set_num_threads(16)

    # speaker = select_speaker_cli(model.speakers)
    speaker = "baya"

    print(f"{selection.lang}/{selection.model}/{speaker}")

    gemini: Chat = _genai_client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=_system_prompts["base"]),
    )

    print("Everything is ready!")

    while True:
        req_bytes = record_audio_to_bytes()

        start_time = time.time()
        req_text = transcribe(req_bytes)
        req_trans_latency = time.time() - start_time
        print("transcription:", req_text)

        print("transcription latency:", req_trans_latency)

        start_time = time.time()
        response = gemini.send_message(req_text)
        resp_gen_latency = time.time() - start_time
        print("response generation latency:", resp_gen_latency)

        start_time = time.time()
        audio: torch.Tensor = model.apply_tts(
            text=response.text,
            speaker=speaker,
            sample_rate=_audio_output_conf["rate"],
            put_accent=True,
        )
        speech_gen_lantency = time.time() - start_time
        print("speech generation latency:", speech_gen_lantency)

        print(
            "overall latency:",
            req_trans_latency + resp_gen_latency + speech_gen_lantency,
        )

        _audio_output.write(audio.numpy().tobytes())


if __name__ == "__main__":
    try:
        main()
    finally:
        _audio_input.close()
        _audio_output.close()
        _audio.terminate()
        _groq_client.close()
