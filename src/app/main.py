import torch
from pynput import keyboard

from audio.recordings import try_end_recording, try_start_recording

wakeup_key = keyboard.Key.f4


def on_key_pressed(key: keyboard.Key) -> None:
    if key == wakeup_key:
        try_start_recording()


def on_key_released(key: keyboard.Key) -> None:
    if key == wakeup_key:
        try_end_recording()


def main() -> None:
    torch._C._jit_set_profiling_mode(False)
    torch.set_num_threads(8)

    with keyboard.Listener(
        on_press=on_key_pressed,
        on_release=on_key_released,
    ) as listener:
        print("Program is started")
        listener.join()


if __name__ == "__main__":
    main()
