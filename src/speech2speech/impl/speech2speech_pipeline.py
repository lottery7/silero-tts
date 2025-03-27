from globals.utils import latency_logging
from speech2text import Speech2TextModel
from text2speech import Text2SpeechModel
from text2text import Text2TextModel

from ..speech2speech import Speech2SpeechModel


class Speech2SpeechPipeline(Speech2SpeechModel):
    def __init__(
        self,
        speech2text_model: Speech2TextModel,
        text2text_model: Text2TextModel,
        text2speech_model: Text2SpeechModel,
    ) -> None:
        self._models: tuple[Speech2TextModel, Text2TextModel, Text2SpeechModel] = (
            speech2text_model,
            text2text_model,
            text2speech_model,
        )

    @latency_logging("Overall latency: {}")
    def generate(self, input_data: bytes) -> bytes:
        input_data: str = self._models[0].generate(input_data)
        input_data = input_data.strip() or "*неразборчиво*"
        input_data: str = self._models[1].generate(input_data)
        input_data: bytes = self._models[2].generate(input_data)

        return input_data
