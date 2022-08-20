from pathlib import Path
from data_io import load_data

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pred_target(pred: np.ndarray, target: np.ndarray, show: bool = True, save_path: Path = None) -> None:
    """plot a 2 dimensional time series signal"""

    plt.figure()
    plt.plot(target, label="target", linestyle="dashed", alpha=0.9)
    plt.plot(pred, label="prediction", linestyle="dashdot", alpha=0.9)
    plt.gca().set_xlabel("sample index")
    plt.gca().set_ylabel("amplitude")
    plt.legend()

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_signal(data: np.ndarray, show: bool = True, save_path: Path = None) -> None:
    """plot a 2 dimensional time series signal"""
    x = list(range(len(data)))
    y = data

    _, ax = plt.subplots()
    plt.plot(x, y)
    ax.set_xlabel("sample index")
    ax.set_ylabel("amplitude")
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_fourier(data: np.ndarray, show: bool = True, save_path: Path = None) -> None:
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

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

def main():
    sampling_rate, data = load_data()
    plot_pred_target(data, data-10)
    plot_signal(data)
    plot_fourier(data)

if __name__ == "__main__":
    main()
