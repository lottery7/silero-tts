from abc import ABC, abstractmethod

import numpy as np


class Text2SpeechModel(ABC):
    @abstractmethod
    def generate(self, text: str) -> tuple[np.ndarray, int]: ...
