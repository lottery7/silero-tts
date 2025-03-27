import re

import torch
from num2words import num2words
from transliterate import translit

from globals.utils import latency_logging

from ..text2speech import Text2SpeechModel


class SileroText2Speech(Text2SpeechModel):
    def __init__(self, language: str, model_id: str, speaker: str) -> None:
        super().__init__()

        self._language = language
        self._model_id = model_id
        self._speaker = speaker

        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self._language,
            speaker=self._model_id,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def _prepare_input(self, input_data: str) -> str:
        # Transliterate text to target language
        input_data = translit(input_data, self._language)

        # Map numbers to words
        input_data = re.sub(
            r"-?[0-9][0-9,._]*",
            lambda match: num2words(
                match.group().replace(",", ".").replace("_", ""),
                lang=self._language,
            ),
            input_data,
        )

        return input_data

    @latency_logging("Silero latency: {}")
    def generate(self, text: str) -> bytes:
        print("Feeding in Silero:", text)

        # Preprocess input
        text = self._prepare_input(text)

        try:
            # Inference model
            audio: torch.Tensor = self._model.apply_tts(
                text=text,
                speaker=self._speaker,
                sample_rate=24000,
                put_yo=True,
                put_accent=True,
            )
            return audio.numpy().tobytes()
        except ValueError:
            return b""
