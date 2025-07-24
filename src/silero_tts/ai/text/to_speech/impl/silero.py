import re

import numpy as np
import torch
from num2words import num2words
from transliterate import translit

from config import CONFIG
from utils import latency_logging

from ..text2speech import Text2SpeechModel


class Silero(Text2SpeechModel):
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

        self._sample_rate = 24000

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
    def generate(self, text: str) -> tuple[np.ndarray, int]:
        print("Feeding in Silero:", text)

        # Preprocess input
        text = self._prepare_input(text)

        try:
            # Inference model
            audio: torch.Tensor = self._model.apply_tts(
                text=text,
                speaker=self._speaker,
                sample_rate=self._sample_rate,
                put_yo=True,
                put_accent=True,
            )
            return audio.numpy(), self._sample_rate
        except ValueError:
            return b""
