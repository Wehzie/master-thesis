from pathlib import Path
import wave

from scipy.io.wavfile import write
from scipy.io.wavfile import read
import numpy as np

DATA_PATH = Path("resources/magpie. 35k, mono, 8-bit, 11025 Hz, 3.3 seconds.wav")

def load_data(data_path: Path = DATA_PATH) -> tuple:
    """
    load a wav file with scipy
    
    return: sampling rate, data
    """
    sampling_rate, data = read(data_path)

    # remove y-offset of unsigned encoding
    if data.dtype == "uint8":
        data = data.astype("int16") - 255 / 2

    return sampling_rate, data
    
def load_data_numpy(data_path: Path = DATA_PATH) -> np.ndarray:
    """load a wav file with stdlib's wave module and numpy"""
    
    # Read file to get buffer                                                                                               
    ifile = wave.open(str(DATA_PATH)) # BUG with pathlib
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)
    rate = ifile.getframerate() # duration of a frame (sample duration)

    # Convert buffer to float32 using NumPy
    audio_as_np_uint8 = np.frombuffer(audio, dtype=np.uint8)
    audio_as_np_float32 = audio_as_np_uint8.astype(np.float32)

    # Normalise float32 array so that values are between 0 and +1.0                                                      
    max_uint8 = 255
    audio_normalised = audio_as_np_float32 / max_uint8

    return audio_normalised

def save_signal_to_wav(data: np.ndarray, path: Path = "data/test.wav") -> None:
    """save a singal to a .wav file"""
    samplerate = 44100 # historical standard
    amplitude = np.iinfo(np.uint8).max # 255
    data *= amplitude # scale
    write(path, samplerate, data.astype(np.uint8))

def main():
    sampling_rate, data = load_data()
    print(f"data: {data}")
    print(f"data length: {len(data)}")
    print(f"sampling rate: {sampling_rate}")
    save_signal_to_wav(data)

if __name__ == "__main__":
    main()
