from abc import ABC, abstractmethod
from typing import Generator

import numpy as np


class Speech2SpeechModel(ABC):
    @abstractmethod
    def generate(self, audio_path: str) -> Generator[np.ndarray, None, None]: ...
