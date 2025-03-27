from abc import ABC, abstractmethod


class Text2SentencesModel(ABC):
    @abstractmethod
    def feed_chunk(self, text_chunk: str) -> list[str]: ...

    @abstractmethod
    def finish(self) -> list[str]: ...

    def feed_full(self, text: str) -> list[str]:
        return self.feed_chunk(text) + self.finish()

