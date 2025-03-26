from abc import ABC, abstractmethod


class Speech2TextModel(ABC):
    @abstractmethod
    def generate(self, input_data: bytes) -> str: ...
