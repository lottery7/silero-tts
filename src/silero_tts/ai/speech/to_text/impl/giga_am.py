import os

from silero_tts.audio.utils import create_wav_from_bytes, split_vad
from silero_tts.utils import latency_logging

from silero_tts..speech2text import Speech2TextModel


class GigaAM(Speech2TextModel):
    MAX_DURATION_S = gigaam.model.LONGFORM_THRESHOLD / gigaam.preprocess.SAMPLE_RATE

    def __init__(self):
        super().__init__()
        self._model: gigaam.GigaAMASR = gigaam.load_model(model_name="ctc")

    def _transcribe(self, audio_path: str) -> str:
        # Inference model
        transcription = self._model.transcribe(audio_path)
        print("Transcription:", transcription)
        return transcription

    def _transcribe_longform(self, audio_path: str) -> str:
        # Split by VAD
        splitted = split_vad(
            audio_path=audio_path,
            max_speech_duration_s=self.MAX_DURATION_S,
        )

        # Transcribe each part and aggregate
        transcriptions = []
        for input_data, sample_rate in splitted:
            if not input_data:
                continue

            va_path = create_wav_from_bytes(
                input_data,
                channels=1,
                sample_width=2,
                sample_rate=sample_rate,
            )

            try:
                transcription = self._transcribe(audio_path=va_path).strip()
                if transcription:
                    transcriptions.append(transcription)
            finally:
                # Delete temporary file
                os.remove(va_path)

        return " ".join(transcriptions)

    @latency_logging("Giga AM latency: {}")
    def generate(self, audio_path: str) -> str:
        try:
            # Transcribe full audio
            transcription = self._transcribe(audio_path=audio_path)
        except:
            # Split by VAD and transcribe each part
            transcription = self._transcribe_longform(audio_path=audio_path)

        transcription = transcription.strip() or "*неразборчиво*"
        print("Full transcription:", transcription)

        return transcription
