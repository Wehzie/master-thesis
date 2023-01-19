from pathlib import Path
from scipy.io.wavfile import read

def load_data(data_path: Path) -> None:
    """
    load a wav file with scipy
    
    return: sampling rate, data, dtype
    """
    sampling_rate, data = read(data_path)
    num_channels = len(data.shape)
    if num_channels == 1:
        number_of_samples = len(data)
    else:
        number_of_samples = len(data[0])
    duration = number_of_samples / sampling_rate
    print(f"loading {data_path}")
    print(f"data shape: {data.shape}")
    print("number of channels: ", num_channels)
    print("number of samples: ", number_of_samples)
    print(f"duration [s]: {duration}")
    print("data type: ", data.dtype)
    print(f"sampling rate: {sampling_rate}")
    
    
data_path = input("type path to .wav file to load: ")
data_path = Path(data_path)
load_data(data_path)
