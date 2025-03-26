import os
import tempfile
import wave

import gigaam

from ..speech2text import Speech2TextModel


class GigaAMSpeech2Text(Speech2TextModel):
    def __init__(self):
        super().__init__()
        self._model: gigaam.GigaAMASR = gigaam.load_model(
            model_name="rnnt",
            fp16_encoder=False,
            device="cpu",
        )

    def generate(self, input_data: bytes) -> str:
        # Create temporary .wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_filename = temp_wav.name

        try:
            # Write input_data to temporary file
            with wave.open(temp_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(input_data)

            # Inference model
            transcription = self._model.transcribe(temp_filename)

            return transcription
        finally:
            # Delete temporary file
            os.remove(temp_filename)
