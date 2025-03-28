from abc import ABC, abstractmethod


class Text2TextModel(ABC):
    @abstractmethod
    def generate(self, input_data: str) -> str: ...
