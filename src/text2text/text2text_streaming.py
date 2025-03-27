from abc import ABC, abstractmethod
from typing import Generator


class Text2TextStreamingModel(ABC):
    @abstractmethod
    def generate_stream(self, input_data: str) -> Generator[str, None, None]: ...
