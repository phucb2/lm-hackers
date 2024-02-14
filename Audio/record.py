import pyaudio
import numpy as np

# Constants
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def process_audio(data):
    # Example: Inverting the audio signal
    processed_data = np.frombuffer(data, dtype=np.int16)
    processed_data = -processed_data
    return processed_data.tobytes()

def audio_callback(in_data, frame_count, time_info, status):
    processed_data = process_audio(in_data)
    return (processed_data, pyaudio.paContinue)

def main():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=audio_callback)

    print("Starting stream...")
    stream.start_stream()

    try:
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        print("Stopping stream...")
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()