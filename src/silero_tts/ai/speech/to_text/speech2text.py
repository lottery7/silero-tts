from abc import ABC, abstractmethod


class Speech2TextModel(ABC):
    @abstractmethod
    def generate(self, audio_path: str) -> str: ...
