from abc import ABC, abstractmethod
from typing import Generator


class Speech2SpeechModel(ABC):
    @abstractmethod
    def generate(self, input_data: bytes) -> Generator[bytes, None, None]: ...
