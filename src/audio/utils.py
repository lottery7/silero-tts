import threading

import pyaudio


def record_pcm_audio(source: pyaudio.Stream) -> bytes:
    # Wait for "Enter" to start recording
    print('Press "Enter" and speak')

    try:
        input()
    except KeyboardInterrupt:
        print("Exit")
        exit(0)

    # Create separate thread for recording
    frames = []
    recording = True

    def record():
        while recording:
            frames.append(source.read(1024))

    thread = threading.Thread(target=record)
    thread.start()

    print('Recording started. Press "Enter" to stop')

    # Wait for "Enter" to stop recording
    input()
    recording = False
    thread.join()

    return b"".join(frames)
