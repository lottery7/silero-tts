from abc import ABC, abstractmethod


class Speech2SpeechModel(ABC):
    @abstractmethod
    def generate(self, input_data: bytes) -> bytes: ...
