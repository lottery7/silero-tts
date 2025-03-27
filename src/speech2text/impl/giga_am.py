import os

import gigaam

from audio.utils import create_wf_from_bytes
from globals.utils import latency_logging

from ..speech2text import Speech2TextModel


class GigaAMSpeech2Text(Speech2TextModel):
    def __init__(self):
        super().__init__()
        self._model: gigaam.GigaAMASR = gigaam.load_model(model_name="ctc")

    @latency_logging("Giga AM latency: {}")
    def generate(self, input_data: bytes) -> str:
        # Create audio file
        audio_path = create_wf_from_bytes(
            input_data,
            channels=1,
            sample_width=2,
            sample_rate=24000,
        )
        try:
            # Inference model
            transcription = self._model.transcribe(audio_path).strip()
            print("Transcription:", transcription)
            return transcription
        finally:
            # Delete audio file
            os.remove(audio_path)
