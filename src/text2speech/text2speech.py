from abc import ABC, abstractmethod


class Text2SpeechModel(ABC):
    @abstractmethod
    def generate(self, text: str) -> bytes: ...
