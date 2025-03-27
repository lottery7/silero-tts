from functools import wraps
from time import time
from typing import Callable


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
