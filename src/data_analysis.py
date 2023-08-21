"""
This module defines metrics and plots for the analysis of signals.

The focus is on plotting one or two signals at a time. 
"""

from typing import Callable, List, Tuple, Union
import data_io
import data_preprocessor
import meta_target

from pathlib import Path
from functools import wraps
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def infer_subplot_rows_cols(n_signals: int) -> Tuple[int, int]:
    """infer the number of rows and columns for a grid of subplots"""
    n_rows = None

    i = 0
    while i**2 <= n_signals:
        n_rows = i
        i += 1
    remainder = n_signals - (i - 1) ** 2
    n_cols = n_rows
    n_rows = n_rows + int(np.ceil(remainder / n_cols))
    return n_rows, n_cols


def plot_multiple_targets_common_axis(
    targets: List[meta_target.MetaTarget], show: bool = False, save_path: Path = None
) -> None:
    """plot multiple targets by aligning to the number of samples in the target with the fewest samples"""
    min_samples = min([target.samples for target in targets])
    time = np.linspace(0, 1, min_samples, endpoint=False)
    target_matrix = np.array([target.signal[:min_samples] for target in targets])
    plot_individual_oscillators(target_matrix, time, show=show, save_path=save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


def plot_multiple_targets(
    targets: List[meta_target.MetaTarget], show: bool = False, save_path: Path = None
) -> None:
    """plot multiple targets in a grid of subplots; plots are aligned on one time axis"""
    nrows, ncols = infer_subplot_rows_cols(len(targets))
    _, axes = plt.subplots(nrows, ncols, layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.plot(targets[i].time, targets[i].signal)
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


def plot_individual_oscillators(
    signal_matrix: np.ndarray,
    time: np.ndarray = None,
    oscillators_per_subplot: Union[None, int] = 25,
    show: bool = False,
    save_path: Path = None,
) -> None:
    """show n individual oscillator signals in a grid of subplots

    args:
        signal_matrix: a matrix of single-oscillator signals
        time: times at which the signals were sampled
        oscillators_per_subplot: number of oscillators per subplot. If None, plot all oscillators in one subplot
        show: show the plot
        save_path: save the plot to a file
    """

    def subset_matrix(signal_matrix: np.ndarray, oscillators_per_subplot: int) -> List[np.ndarray]:
        """split a matrix into subsets of n rows, returns a view"""
        subsets = []
        for row in range(0, signal_matrix.shape[0], oscillators_per_subplot):
            subsets.append(signal_matrix[row : row + oscillators_per_subplot, :])
        return subsets

    def plot_row_per_plot(signal_matrix: np.ndarray, axes: plt.Axes) -> None:
        """plot one signal per row of a matrix in a grid of subplots"""
        n_signals = signal_matrix.shape[0]  # one signal per row
        n_rows, n_cols = infer_subplot_rows_cols(n_signals)

        # plot one signal into each subplot
        signal_counter = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if signal_counter >= n_signals:
                    break
                if time is None:
                    axes[r, c].plot(signal_matrix[signal_counter, :])
                else:
                    axes[r, c].plot(time, signal_matrix[signal_counter, :])
                signal_counter += 1

    def init_axes(signal_matrix: np.ndarray) -> plt.Axes:
        """initialize a grid of subplots"""
        n_signals = signal_matrix.shape[0]
        rows, cols = infer_subplot_rows_cols(n_signals)
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
        return fig, axes

    def add_labels(axes: plt.Axes, signal_matrix: np.ndarray) -> None:
        """add labels to the subplots"""
        n_signals = signal_matrix.shape[0]
        rows, cols = infer_subplot_rows_cols(n_signals)
        plt.suptitle("individual oscillators")
        if time is None:  # samples
            axes[rows - 1, cols // 2].set_xlabel("time [a.u.]")
        else:  # time
            axes[rows - 1, cols // 2].set_xlabel("time [s]")
        axes[rows // 2, 0].set_ylabel("amplitude [a.u.]")

    figures = []
    if oscillators_per_subplot is None:  # plot all oscillators in one figure
        fig, axes = init_axes(signal_matrix)
        figures.append(fig)
        plot_row_per_plot(signal_matrix, axes)
        add_labels(axes, signal_matrix)

    else:  # divide oscillators across figures
        subsets = subset_matrix(signal_matrix, oscillators_per_subplot)
        for i, subset in enumerate(subsets):
            fig, axes = init_axes(subset)
            figures.append(fig)
            plot_row_per_plot(subset, axes)
            add_labels(axes, subset)

    if save_path:  # save all figures generated in this function
        for i, fig in enumerate(figures):
            path = (
                None
                if save_path is None
                else Path(save_path.parent, f"{save_path.stem}_{i}{save_path.suffix}")
            )
            fig.savefig(path, dpi=300)
    if show:
        plt.show()


def plot_f0_hist(
    signal_matrix: np.ndarray,
    sample_spacing: float,
    title: str = None,
    show: bool = False,
    save_path: Path = None,
) -> None:
    """plot a histogram of the fundamental frequency of each oscillator in the matrix"""
    f0_li = []
    for i in range(signal_matrix.shape[0]):
        f0_li.append(get_freq_from_fft_v2(signal_matrix[i, :], sample_spacing))

    plt.figure()
    plt.hist(f0_li, bins="auto")
    plt.gca().set_xlabel("f0 [Hz]")
    plt.gca().set_ylabel("count")

    if title:
        plt.title(title)
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_weight_hist(
    weights: np.ndarray, show: bool = False, save_path: Path = None, title: str = None
) -> None:
    """plot a histogram to visualize the distribution of weights in an oscillator ensemble

    args:
        weights: a vector of weights
        show: show the plot
        save_path: save the plot to the provided path
        title: title of the plot
    """
    plt.figure()
    plt.hist(weights, bins="auto")
    plt.gca().set_xlabel("weight")
    plt.gca().set_ylabel("count")
    if title:
        plt.title(title)

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_rmse_hist(
    rmse_li: list, show: bool = False, title: str = None, save_path: Path = None
) -> None:
    """produce a histogram over n samples"""
    plt.figure()
    plt.hist(rmse_li, bins=len(rmse_li) // 5)
    plt.gca().set_xlabel("rmse")
    plt.gca().set_ylabel("count")
    if title:
        plt.title(title)

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_pred_target(
    pred: np.ndarray,
    target: np.ndarray,
    time: Union[np.ndarray, None] = None,
    show: bool = False,
    save_path: Path = None,
    title: str = None,
    ylabel="amplitude [a.u.]",
    large_font: bool = True,
) -> None:
    """plot a 2 dimensional time series signal

    args:
        pred: the approximation (prediction) to the target signal
        target: the target signal to be approximated
        time: the time axis shared by both signals
        show: show the plot
        save_path: save the plot to the provided path
        title: title of the plot
        ylabel: label of the y axis
        large_font: if True, increase font size
    """
    fig, ax1 = plt.subplots()  # ax1 is on the bottom

    if time is not None:  # time axis
        ax2 = ax1.twiny()  # ax2 is on the top

        # assign which axis goes top or bottom
        ax_time = ax1
        ax_samples = ax2

        # time axis
        ax_time.plot(time, target, label="target", linestyle="dashed", alpha=0.9)
        ax_time.plot(time, pred, label="prediction", linestyle="dashdot", alpha=0.9)
        ax_time.set_xlabel("time [s]")
        ax_time.set_ylabel(ylabel)

        # force scientific notation for tick labels, useful for low freq targets
        # ax_time.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        if large_font:
            ax_time.set_xlabel("time [s]", fontsize=18)
            ax_time.set_ylabel(ylabel, fontsize=18)
            ax_time.tick_params(axis="both", which="major", labelsize=18)
            ax_time.tick_params(axis="both", which="minor", labelsize=18)
            exponent_x = ax_time.xaxis.get_offset_text()
            exponent_x.set_size(18)
            exponent_y = ax_time.yaxis.get_offset_text()
            exponent_y.set_size(18)
    else:
        ax_samples = ax1

    # sample axis
    ax_samples.plot(target, label="target", linestyle="dashed", alpha=0.9)
    ax_samples.plot(pred, label="prediction", linestyle="dashdot", alpha=0.9)
    ax_samples.set_ylabel(ylabel)
    if time is None:
        ax_samples.set_xlabel("time [a.u.]")
    else:
        # hide sample ticks
        ax_samples.get_xaxis().set_visible(False)
        # add sampling rate to title
        duration = time[-1] - time[0]
        samples = len(time)
        sampling_rate = int(1 / (duration / samples))
        title += r"; $f_s$=" + f"{sampling_rate} [Hz]"
    plt.legend()
    if large_font:
        plt.legend(fontsize=18)
    if title:
        plt.title(title)

    fig.tight_layout()

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_signal(
    y: np.ndarray,
    x_time: np.ndarray = None,
    sampling_rate: int = None,
    ylabel: str = None,
    title: str = "",
    show: bool = False,
    save_path: Path = None,
) -> None:
    """plot a 1d time series"""
    _, ax1 = plt.subplots()

    if x_time is not None:
        ax2 = ax1.twiny()

        # assign which axis goes top or bottom
        ax_time = ax1

        # hide samples axis
        ax_samples = ax2
        ax_samples.get_xaxis().set_visible(False)

        ax_time.plot(x_time, y)
        ax_time.set_xlabel("time [s]")
    else:  # no time axis
        ax_samples = ax1

    # add ylabel
    if ylabel is None:
        ax1.set_ylabel("amplitude [a.u.]")
    else:
        ax1.set_ylabel(ylabel)

    # add sampling rate to title
    out_title = ""
    if len(title) > 0:
        out_title = title + ", "
    if sampling_rate is not None:
        out_title += r"$f_s$=" + f"{sampling_rate} [Hz]"
    else:
        out_title += f"{len(y)} samples"
    plt.title(out_title)

    ax_samples.set_xlabel("time [a.u.]")
    ax_samples.plot(y)

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_fourier(
    data: np.ndarray, title: str = None, show: bool = False, save_path: Path = None
) -> None:
    """plot the fourier transform of a 2 dimensional time series"""
    # apply fourier transform to signal
    # can use rfft since data purely real
    # twice as fast as fft using complex conjugates
    spectrum = np.fft.rfft(data)
    abs_spec = abs(spectrum)  # absolute spectrum
    sample_spacing = 1.0 / 44100  # inverse of the sampling rate
    freq = np.fft.fftfreq(len(abs_spec), d=sample_spacing)

    # plot negative spectrum first and avoid horizontal line in plot
    abs_spec = np.fft.fftshift(abs_spec)
    freq = np.fft.fftshift(freq)

    _, ax = plt.subplots()
    plt.plot(freq, abs_spec)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("log-frequency [Hz]")
    ax.set_ylabel("log-amplitude")

    if title:
        plt.title(title)
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)


def compute_rmse(p: np.ndarray, t: np.ndarray, verbose: bool = False, pad: bool = False) -> float:
    """Compute root mean square error (RMSE) between prediction and target signal."""
    if pad:
        p, t = data_preprocessor.align_signals(p, t)
    rmse = np.sqrt(((p - t) ** 2).mean())
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
    freqs = np.fft.rfftfreq(len(s), sample_spacing)  # frequency axis
    abs_spec = abs(spectrum)  # absolute spectrum
    # choose the largest peak in the spectrum
    # for example
    #   return freqs[abs_spec.argmax()]
    # however the following seems more robust
    # see, https://stackoverflow.com/questions/59265603/how-to-find-period-of-signal-autocorrelation-vs-fast-fourier-transform-vs-power
    inflection = np.diff(np.sign(np.diff(abs_spec)))
    peaks = (inflection < 0).nonzero()[0] + 1
    peak = peaks[abs_spec[peaks].argmax()]
    return freqs[peak]


def get_freq_from_fft_v2(s: np.ndarray, sample_spacing: float) -> float:
    """
    compute fundamental frequency of an oscillator using FFT.

    compared to v1, this version seems more numerically stable
    """
    # apply fourier transform to signal
    spectrum = np.fft.fft(s)
    abs_spec = abs(spectrum)
    freq = np.fft.fftfreq(len(abs_spec), d=sample_spacing)

    # compute fundamental frequency
    # usually we want f1, and usually -f0=f1
    # but sometimes when frequency is low and sampling rate high, f1 yields 0
    # for example, sampling_rate = 10001, frequency = 300
    nlargest = pd.Series(abs_spec).nlargest(2)
    nlargest_arg = nlargest.index.values.tolist()
    f0, f1 = freq[nlargest_arg[0]], freq[nlargest_arg[1]]
    return abs(max(f0, f1, key=abs))


def get_max_freq_from_fft(s: np.ndarray, sample_spacing: float) -> float:
    """find the fastest frequency component in a signal using FFT"""
    freqs = np.fft.rfftfreq(len(s), sample_spacing)  # frequency axis
    return freqs[-1]


def main():
    sampling_rate, data, _ = data_io.load_data()
    plot_pred_target(data, data - 10)
    plot_signal(data, show=True)
    plot_fourier(data, show=True)


if __name__ == "__main__":
    main()
