

from data_io import load_data, save_signal_to_wav
from data_analysis import plot_signal, plot_fourier

from scipy import signal
import numpy as np

def scale_down(data: np.ndarray, s_factor: float = 0.01) -> np.ndarray:
    """reduce the size of a signal by a given factor via downsampling"""
    return signal.resample(data, int(len(data) * s_factor))


def pad_zero(short: np.ndarray, len_long: int) -> np.ndarray:
    """evenly zero-pad a short signal up to the desired length"""
    # evenly pad with zeros
    to_pad = len_long - len(short)

    # deal with an odd number of padding so that dimensions are exact
    to_pad_odd = 0
    if to_pad % 2 == 1:
        to_pad_odd = 1

    padded = np.pad(short, (to_pad//2, to_pad//2 + to_pad_odd), mode="constant")
    return padded


def align_signals(p: np.ndarray, t: np.ndarray) -> tuple:
    """take two different length signals and return them with the same length"""
    if len(p) < len(t):
        return pad_zero(p, len(t)), t
    elif len(t) < len(p):
        return p, pad_zero(t, len(p))
    else:
        return p, t

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