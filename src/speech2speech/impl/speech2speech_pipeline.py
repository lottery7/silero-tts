from typing import Generator

from speech2text import Speech2TextModel
from text2sentences import Text2SentencesModel
from text2speech import Text2SpeechModel
from text2text import Text2TextModel

from ..speech2speech import Speech2SpeechModel


class Speech2SpeechPipeline(Speech2SpeechModel):
    def __init__(
        self,
        speech2text_model: Speech2TextModel,
        text2text_model: Text2TextModel,
        text2sentences_model: Text2SentencesModel,
        text2speech_model: Text2SpeechModel,
    ) -> None:
        self._models: tuple[
            Speech2TextModel,
            Text2TextModel,
            Text2SentencesModel,
            Text2SpeechModel,
        ] = (
            speech2text_model,
            text2text_model,
            text2sentences_model,
            text2speech_model,
        )

    def generate(self, input_data: bytes) -> Generator[bytes, None, None]:
        input_data: str = self._models[0].generate(input_data)

        generated = self._models[1].generate(input_data)
        sentences = self._models[2].feed_full(generated)

        for sentence in sentences:
            yield self._models[3].generate(sentence)
