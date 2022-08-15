

from data_io import load_data, save_signal_to_wav
from data_analysis import plot_signal, plot_fourier

from scipy import signal
import numpy as np

def scale_down(data: np.ndarray, s_factor: float = 0.01) -> np.ndarray:
    """reduce the size of a signal by a given factor via downsampling"""
    return signal.resample(data, int(len(data) * s_factor))

def main():
    sampling_rate, data = load_data()
    plot_signal(data)
    plot_fourier(data)

    sd_data = scale_down(data)
    
    plot_signal(sd_data)
    plot_fourier(sd_data)

    save_signal_to_wav(sd_data)

if __name__ == "__main__":
    main()