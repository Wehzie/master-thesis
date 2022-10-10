from typing import Callable
import data_io
import data_preprocessor

from pathlib import Path
from functools import wraps
import time

import numpy as np
import matplotlib.pyplot as plt

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


def plot_pred_target(pred: np.ndarray, target: np.ndarray, show: bool = False,
    save_path: Path = None, title: str = None) -> None:
    """plot a 2 dimensional time series signal"""

    plt.figure()
    plt.plot(target, label="target", linestyle="dashed", alpha=0.9)
    plt.plot(pred, label="prediction", linestyle="dashdot", alpha=0.9)
    plt.gca().set_xlabel("sample index")
    plt.gca().set_ylabel("amplitude")
    plt.legend()
    if title: plt.title(title)

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_signal(y: np.ndarray, x: np.ndarray = None, ylabel: str = None, title: str = None, show: bool = False, save_path: Path = None) -> None:
    """plot a 2 dimensional time series signal"""
    _, ax = plt.subplots()

    if x is None:
        ax.set_xlabel("sample index")
        x = list(range(len(y)))
    else:
        ax.set_xlabel("time")

    if ylabel is None:
        ax.set_ylabel("amplitude")
    else:
        ax.set_ylabel(ylabel)

    if title: plt.title(title)
    
    plt.plot(x, y)
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
    @wraps(func)
    def wrap(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        time_elapsed = time.time() - t0
        print(f"time elapsed: {time_elapsed:.2f} s")
        print(f"func:{func.__name__}")
        print(f"args:{args}")
        print(f"kwargs:{kwargs}")
        return result
    return wrap

def main():
    sampling_rate, data = data_io.load_data()
    plot_pred_target(data, data-10)
    plot_signal(data)
    plot_fourier(data)

if __name__ == "__main__":
    main()
