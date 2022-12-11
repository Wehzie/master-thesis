from typing import Callable, Tuple, Union
import data_io
import data_preprocessor

from pathlib import Path
from functools import wraps
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_n(data: np.ndarray, show: bool = True, save_path: Path = None) -> None:
    """plot n signals in a single plot"""

    # generate a grid of subplots
    n_signals = data.shape[0] # rows
    n_rows = None
    
    i = 0
    while i**2 <= n_signals:
        n_rows = i
        i += 1
    remainder = n_signals - (i-1)**2
    n_cols = n_rows
    n_rows = n_rows + int(np.ceil(remainder / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)

    # plot signals into each subplot
    idx_signal = 0
    print("rows, cols")
    print(n_rows, n_cols)
    for r in range(n_rows):
        for c in range(n_cols):
            if idx_signal >= n_signals: break
            ax[r, c].plot(data[idx_signal, :])
            idx_signal += 1

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def hist_rmse(rmse_li: list, show: bool = False, title: str = None, save_path: Path = None) -> None:
    """produce a histogram over n samples"""
    plt.figure()
    plt.hist(rmse_li, bins=len(rmse_li)//10)
    plt.gca().set_xlabel("rmse")
    plt.gca().set_ylabel("count")
    if title: plt.title(title)

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_pred_target(pred: np.ndarray, target: np.ndarray, time: Union[np.ndarray, None] = None,
    show: bool = False, save_path: Path = None, title: str = None) -> None:
    """plot a 2 dimensional time series signal"""
    fig, ax1 = plt.subplots()

    if time is not None: # time axis
        ax2 = ax1.twiny()
        ax2.plot(time, target, label="target", linestyle="dashed", alpha=0.9)
        ax2.plot(time, pred, label="prediction", linestyle="dashdot", alpha=0.9)
        ax2.set_xlabel("time [s]")

    # sample axis
    ax1.plot(target, label="target", linestyle="dashed", alpha=0.9)
    ax1.plot(pred, label="prediction", linestyle="dashdot", alpha=0.9)
    ax1.set_xlabel("sample index")
    ax1.set_ylabel("amplitude")
    plt.legend()
    if title: plt.title(title)

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_signal(y: np.ndarray, x_time: np.ndarray = None, ylabel: str = None, title: str = None, show: bool = False, save_path: Path = None) -> None:
    """plot a 2 dimensional time series signal"""
    fig, ax1 = plt.subplots()

    if x_time is not None:
        ax2 = ax1.twiny()
        ax2.plot(x_time, y)
        ax2.set_xlabel("time [s]")

    if ylabel is None:
        ax1.set_ylabel("amplitude")
    else:
        ax1.set_ylabel(ylabel)

    if title: plt.title(title)
    
    ax1.set_xlabel("sample index")
    x_samples = list(range(len(y)))
    ax1.plot(x_samples, y)

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_fourier(data: np.ndarray, title: str = None, show: bool = False, save_path: Path = None) -> None:
    """plot the fourier transform of a 2 dimensional time series"""
    # apply fourier transform to signal
	# can use rfft since data purely real
    # twice as fast as fft using complex conjugates
    spectrum = np.fft.rfft(data)
    abs_spec = abs(spectrum) # absolute spectrum
    sample_spacing = 1.0 / 44100 # inverse of the sampling rate
    freq = np.fft.fftfreq(len(abs_spec), d=sample_spacing)
	
	# plot negative spectrum first and avoid horizontal line in plot
    abs_spec = np.fft.fftshift(abs_spec)
    freq = np.fft.fftshift(freq)

    _, ax = plt.subplots()
    plt.plot(freq, abs_spec)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("log-frequency [Hz]")
    ax.set_ylabel('log-amplitude')
    
    if title:
        plt.title(title)
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

        
def compute_rmse(p: np.ndarray, t: np.ndarray, verbose: bool = False, pad: bool = False) -> float:
    """
    Compute root mean square error (RMSE) between prediction and target signal.
    """
    if pad: p, t = data_preprocessor.align_signals(p, t)
    rmse = np.sqrt(((p-t)**2).mean())
    if verbose:
        print(f"RMSE: {rmse}")
    return rmse


def print_time(func: Callable) -> Callable:
    """print the time a callable took after execution"""
    @wraps(func)
    def wrap(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        time_elapsed = time.time() - t0
        print(f"\ntime elapsed for func '{func.__name__}': {time_elapsed:.2f} s\n")
        return result
    return wrap


def get_freq_from_fft(s: np.ndarray, sample_spacing: float) -> float:
    """compute fundamental frequency of an oscillator using FFT"""
    spectrum = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(len(s), sample_spacing) # frequency axis
    abs_spec = abs(spectrum) # absolute spectrum
    # choose the largest peak in the spectrum
    # for example
    #   return freqs[abs_spec.argmax()]
    # however the following seems more robust
    # see, https://stackoverflow.com/questions/59265603/how-to-find-period-of-signal-autocorrelation-vs-fast-fourier-transform-vs-power
    inflection = np.diff(np.sign(np.diff(abs_spec)))
    peaks = (inflection < 0).nonzero()[0] + 1
    peak = peaks[abs_spec[peaks].argmax()]
    return freqs[peak]

def get_freq_from_fft_v2(s: np.ndarray, sample_spacing: float) -> Tuple:
    """compute fundamental frequency of an oscillator using FFT,
    compared to v1, this version seems more numerically stable"""
    # apply fourier transform to signal
    spectrum = np.fft.fft(s)
    abs_spec = abs(spectrum)
    freq = np.fft.fftfreq(len(abs_spec), d=sample_spacing)

    # compute fundamental frequency
    nlargest = pd.Series(abs_spec).nlargest(2)
    nlargest_arg = nlargest.index.values.tolist()
    return abs(freq[nlargest_arg[1]])

def main():
    sampling_rate, data, _ = data_io.load_data()
    plot_pred_target(data, data-10)
    plot_signal(data, show=True)
    plot_fourier(data, show=True)

if __name__ == "__main__":
    main()
