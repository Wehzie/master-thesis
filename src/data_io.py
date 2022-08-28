import glob
import json
from pathlib import Path
import wave

from scipy.io.wavfile import write
from scipy.io.wavfile import read
import numpy as np
import pandas as pd

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

def find_dir_name(parent_path: Path) -> Path:
    """find unused directory name for a search experiment and make directory"""
    for i in range(100):
        path = Path(parent_path / f"experiment{i}")
        if not path.exists():
            path.mkdir()
            break
    return path

def json_to_df(path: Path) -> pd.DataFrame:
    """aggregate parameters and results of multiple json files into a dataframe"""
    data_rows = []
    for json_file in glob.glob(str(path) + "/*.json"):
        with open(json_file) as f:
            data_rows.append(json.load(f))
    df = pd.DataFrame(data_rows)
    return df

def load_sim_data(data_path: Path) -> pd.DataFrame:
    """load the simulation data written to file by ngspice into python"""
    df = pd.DataFrame()
    df = pd.read_csv(data_path, sep="[ ]+", engine="python") # match any number of spaces
    return df

def main():
    sampling_rate, data = load_data()
    print(f"data: {data}")
    print(f"data length: {len(data)}")
    print(f"sampling rate: {sampling_rate}")
    save_signal_to_wav(data)

if __name__ == "__main__":
    main()
