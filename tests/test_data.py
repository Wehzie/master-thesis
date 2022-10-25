from pathlib import Path
import sys
import os

# add code in src to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from data_io import *

def test_signal_to_wav() -> None:
    """test loading from wav, then saving, then loading again"""
    
    # load default data
    sampling_rate, data, data.dtype = load_data()
    # save default data to wav
    save_signal_to_wav(data, samplerate=sampling_rate, dtype=data.dtype, path=Path("data/test.wav"))
    # load from wav
    sampling_rate, data_from_wav, data.dtype = load_data(Path("data/test.wav"))
