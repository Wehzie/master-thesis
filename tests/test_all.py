from pathlib import Path
import sys
import os

# add code in src to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import data_io as io
import data_analysis as analysis
import data_preprocessor as preprocessor

import numpy as np

def test_signal_to_wav(visual: bool = False) -> None:
    """test by loading from wav, then saving, then loading again"""
    from data_analysis import plot_signal
    
    data = io.load_data() # load default data
    preprocessor.save_signal_to_wav(data) # save default data to wav
    data_from_wav = io.load_data(Path("data/test.wav")) # load from wav
    
    if visual:
        plot_signal(data) # plot default data
        plot_signal(data_from_wav)
    
    # compare default and from_wav data
    assert np.array_equal(data, data_from_wav), "Data from wav files is not identical"
