import os
import threading
from queue import Empty, Queue
from typing import Generator

import numpy as np

from silero_tts.ai.speech.to_speech.speech2speech import Speech2SpeechModel
from silero_tts.ai.speech.to_text import Speech2TextModel
from silero_tts.ai.text.to_speech import Text2SpeechModel
from silero_tts.ai.text.to_text import Text2TextModel
from silero_tts.audio.utils import create_wav_from_bytes, resample
from silero_tts.config import CONFIG
from silero_tts.utils import split_by_sentences_ru


def _process_sentences(
    model: Text2SpeechModel,
    sentences: list[str],
    queue: Queue,
) -> None:
    for s in sentences:
        audio_data, sample_rate = model.generate(s)
        if sample_rate != CONFIG.audio.output.sample_rate:
            audio_data = resample(
                audio_data,
                base_sr=sample_rate,
                target_sr=CONFIG.audio.output.sample_rate,
            )
        audio_data = (audio_data * 32767).astype(np.int16).tobytes()
        queue.put_nowait(audio_data)


class Speech2SpeechPipeline(Speech2SpeechModel):
    def __init__(
        self,
        speech2text_model: Speech2TextModel,
        text2text_model: Text2TextModel,
        text2speech_model: Text2SpeechModel,
    ) -> None:
        self._models: tuple[
            Speech2TextModel,
            Text2TextModel,
            Text2SpeechModel,
        ] = (
            speech2text_model,
            text2text_model,
            text2speech_model,
        )

    def generate(self, input_data: bytes) -> Generator[np.ndarray, None, None]:
        # Inference Speech To Text model
        # Create input audio file
        wf = create_wav_from_bytes(
            input_data=input_data,
            channels=CONFIG.audio.input.channels,
            sample_width=2,
            sample_rate=CONFIG.audio.input.sample_rate,
        )
        try:
            input_data = self._models[0].generate(audio_path=wf)
        finally:
            # Delete wav file
            os.remove(wf)

        # Inference Text To Text model
        generated = self._models[1].generate(input_data)

        # Split response by sentences
        sentences = split_by_sentences_ru(generated)

        # Generate speech

        # We have to process text sentence-by-sentence. If we do it with yield in the same thread,
        # we would inference model each time next() is called on the generator. This is why we are
        # creating another thread.
        queue = Queue()
        thread = threading.Thread(
            target=_process_sentences,
            args=[self._models[2], sentences, queue],
            daemon=True,
        )
        thread.start()

        for i in range(len(sentences)):
            try:
                yield queue.get(timeout=10)
            except Empty:
                if not thread.is_alive():
                    break
