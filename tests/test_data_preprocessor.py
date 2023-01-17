# add code in src to path
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import data_io
import data_analysis
import data_preprocessor

def test_resample_by_factor():
    sampling_rate, data = data_io.load_data()
    data_analysis.plot_signal(data)
    data_analysis.plot_fourier(data)

    sd_data = data_preprocessor.resample_by_factor(data)
    
    data_analysis.plot_signal(sd_data)
    data_analysis.plot_fourier(sd_data)

    data_io.save_signal_to_wav(sd_data)