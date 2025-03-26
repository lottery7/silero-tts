from abc import ABC, abstractmethod


class Text2SpeechModel(ABC):
    @abstractmethod
    def generate(self, input_data: str) -> bytes: ...
