from functools import wraps
from time import time
from typing import Callable

import spacy
import torch


def latency_logging(_format: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()

            return_val = func(*args, **kwargs)

            end_time = time()
            latency = end_time - start_time
            print(_format.format(latency))

            return return_val

        return wrapper

    return decorator


def torch_single_thread(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return func(*args, **kwargs)
        finally:
            torch.set_num_threads(num_threads)

    return wrapper


_nlp = spacy.load("ru_core_news_sm")


def split_by_sentences_ru(text: str) -> list[str]:
    doc = _nlp(text)
    return [sent.text.strip() for sent in doc.sents]
