import spacy

from globals.utils import latency_logging

from ..text2sentences import Text2SentencesModel


class SpacySentenceBuffer(Text2SentencesModel):
    def __init__(self, model: str) -> None:
        super().__init__()
        self._model = spacy.load(model)
        self._buffer = ""

    def _split_buffer(self) -> list[str]:
        print("Buffer:", self._buffer)

        doc = self._model(self._buffer)
        sentences = [sent.text.strip() for sent in doc.sents]
        print("Sentences:", sentences)

        return sentences

    def feed_chunk(self, text_chunk: str) -> list[str]:
        self._buffer += text_chunk
        sentences = self._split_buffer()

        if not sentences:
            self._buffer = ""
            return []

        self._buffer = sentences[-1]
        return sentences[:-1]

    def finish(self) -> list[str]:
        sentences = self._split_buffer()
        self._buffer = ""
        return sentences

    @latency_logging("Spacy latency: {}")
    def feed_full(self, text: str) -> list[str]:
        self._buffer = text
        return self._split_buffer()
