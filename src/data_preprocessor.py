
import data_io
import data_analysis

from scipy import signal
import numpy as np
import numpy.typing as npt
import pandas as pd

def sample_down(data: np.ndarray, s_factor: float = 0.01) -> npt.NDArray[np.float64]:
    """reduce the size of a signal by a given factor via downsampling"""
    return signal.resample(data, int(len(data) * s_factor))

def sample_down_int(data: np.ndarray, s_factor: float) -> np.ndarray:
    """downsample a signal by taking each n-th sample.
    input signal maintains its type"""
    step = int(1/s_factor)
    return data[0:-1:step]

def scale_up(short: np.ndarray, len_long: int) -> np.ndarray:
    """scale a short signal up to the desired length by repeating each symbol in-place"""
    short = pd.Series(short)
    short = short.repeat(len_long//len(short))
    # pad with zeros at the end
    to_pad = len_long - len(short)
    padded = pad_zero(short, len_long)
    return padded


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

def clean_signal(s: np.ndarray, points_dropped: int = 200) -> np.ndarray:
    """remove startup and y-offset"""
    no_startup = s[points_dropped:]
    no_offset = no_startup - min(no_startup)
    return no_offset

def take_middle_third(signal: np.ndarray) -> np.ndarray:
    """return only the middle third of a signal"""
    third = len(signal)//3
    return signal[third:len(signal)-third]

def norm1d(signal: np.ndarray) -> np.ndarray:
    """normalize a 1d-signal to the range from 0 to 1"""
    return np.divide(signal, np.max(np.abs(signal), axis=0))

def norm2d(matrix: np.ndarray) -> np.ndarray:
    """normalize a each row in a matrix to the range from 0 to 1"""
    return np.apply_along_axis(norm1d, 0, matrix)

def main():
    sampling_rate, data = data_io.load_data()
    data_analysis.plot_signal(data)
    data_analysis.plot_fourier(data)

    sd_data = sample_down(data)
    
    data_analysis.plot_signal(sd_data)
    data_analysis.plot_fourier(sd_data)

    data_io.save_signal_to_wav(sd_data)

if __name__ == "__main__":
    main()