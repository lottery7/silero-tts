import os
import threading
from typing import Any

import gigaam

from audio.utils import create_wf_from_bytes
from globals.utils import latency_logging

from ..speech2text import Speech2TextModel


class GigaAMSpeech2Text(Speech2TextModel):
    def __init__(self):
        super().__init__()
        self._model: gigaam.GigaAMASR = gigaam.load_model(model_name="ctc")
        self._emo_model: gigaam.GigaAMEmo = gigaam.load_model(model_name="emo")

    def _transcribe_task(self, audio_path: str, result_container: list[Any]) -> None:
        # Inference ASR model
        transcription = self._model.transcribe(audio_path).strip() or "*неразборчиво*"
        result_container.append(transcription)

        print("Transcription:", transcription)

    def _classify_emo_task(self, audio_path: str, result_container: list[Any]) -> None:
        # Inference EMO model
        emotion2prob = self._emo_model.get_probs(audio_path)
        emotion = max(emotion2prob, key=emotion2prob.get)
        result_container.append(emotion)

        print("Emotion:", emotion)

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
            # Run tasks in parallel
            transcription = []
            emotion = []

            transcription_thread = threading.Thread(
                target=self._transcribe_task,
                args=[audio_path, transcription],
            )

            classification_thread = threading.Thread(
                target=self._classify_emo_task,
                args=[audio_path, emotion],
            )

            transcription_thread.start()
            classification_thread.start()

            transcription_thread.join()
            classification_thread.join()

            transcription = transcription[0]
            emotion = emotion[0]

            # Aggregate and return results
            result = f"Эмоция пользователя: {emotion}\nТранскрипция голоса пользователя: {transcription}"

            return result
        finally:
            # Delete audio file
            os.remove(audio_path)
