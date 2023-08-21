"""
This module applies preprocessing to signals.

In particular it implements resampling, peak detection and padding functions.
"""

from typing import Union, Tuple

from scipy import signal
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def resample(data: np.ndarray, samples: int) -> np.ndarray:
    """modify the size of a signal to the desired number of samples"""
    return signal.resample(data, samples, axis=0)


def downsample_typesafe(data: np.ndarray, samples: int) -> np.ndarray:
    """reduce the size of a signal to the desired number of samples while maintaining the type

    args:
        data: the signal to be resampled
        samples: the desired number of samples

    returns:
        the resampled signal
    """
    s_factor = samples / len(data)
    step = np.floor(1 / s_factor).astype(int)
    resampled = data[0:-1:step]
    guard = resampled[0:samples]  # make sure the signal is exactly the desired length
    # experimentally, it seems at most 1 sample is cut off by the guard
    assert len(guard) == samples
    return guard


def resample_by_factor(data: np.ndarray, s_factor: float = 0.01) -> npt.NDArray[np.float64]:
    """resample a signal obtaining a length of len(data)*s_factor"""
    return signal.resample(data, int(len(data) * s_factor))


def downsample_by_factor_typesafe(data: np.ndarray, s_factor: float) -> np.ndarray:
    """downsample a signal by taking each n-th sample, input signal maintains its type"""
    step = int(1 / s_factor)
    return data[0:-1:step]


def get_sampling_rate_after_resample(
    old_signal: np.ndarray, new_signal: np.ndarray, old_sampling_rate: int
) -> int:
    """compute the sampling rate after resampling a signal"""
    return int(new_signal.size / old_signal.size * old_sampling_rate)


def scale_up(short: np.ndarray, len_long: int) -> np.ndarray:
    """scale a short signal up to the desired length by repeating each symbol in-place"""
    short = pd.Series(short)
    short = short.repeat(len_long // len(short))
    # pad with zeros at the end
    padded = pad_zero(short, len_long)
    return padded


def interpolate_sinc_time(signal: np.ndarray, time: np.ndarray, time_new: np.ndarray) -> np.ndarray:
    """interpolate a signal using sinc (sinus cardinalis, Whittaker-Shannon) interpolation

    args:
        signal: the signal to be interpolated
        time: the time points of the signal
        time_new: the time points of the interpolated signal

    references:
        - https://gist.github.com/endolith/1297227
    """
    if len(signal) != len(time):
        raise ValueError("time and signal must have the same length")

    T = time[1] - time[0]  # period/signal spacing

    sincM = np.tile(time_new, (len(time), 1)) - np.tile(time[:, np.newaxis], (1, len(time_new)))
    signal_new = np.dot(signal, np.sinc(sincM / T))
    return signal_new


def interpolate_sinc_sampling_rate(
    signal: np.ndarray, old_sampling_rate: int, new_sampling_rate: int
) -> np.ndarray:
    """interpolate a signal using sinc (sinus cardinalis, Whittaker–Shannon) interpolation"""
    if new_sampling_rate < old_sampling_rate:
        raise ValueError("interpolation requires the new sampling rate to be larger")

    duration = len(signal) / old_sampling_rate
    old_time = np.linspace(0, duration, len(signal), endpoint=False)

    new_samples = np.round(duration * new_sampling_rate).astype(int)
    new_time = np.linspace(0, duration, new_samples)

    T = old_time[1] - old_time[0]  # period/signal spacing

    sincM = np.tile(new_time, (len(old_time), 1)) - np.tile(
        old_time[:, np.newaxis], (1, len(new_time))
    )
    signal_new = np.dot(signal, np.sinc(sincM / T))
    return signal_new


def pad_zero(short: np.ndarray, len_long: int) -> np.ndarray:
    """evenly zero-pad a short signal up to the desired length"""
    # evenly pad with zeros
    to_pad = len_long - len(short)

    # deal with an odd number of padding so that dimensions are exact
    to_pad_odd = 0
    if to_pad % 2 == 1:
        to_pad_odd = 1

    padded = np.pad(short, (to_pad // 2, to_pad // 2 + to_pad_odd), mode="constant")
    return padded


def align_signals(p: np.ndarray, t: np.ndarray) -> tuple:
    """take two different length signals and return them with the same length"""
    if len(p) < len(t):
        return pad_zero(p, len(t)), t
    elif len(t) < len(p):
        return p, pad_zero(t, len(p))
    else:
        return p, t


def align_signals_cut(p: np.ndarray, t: np.ndarray) -> tuple:
    """take two different length signals and return them with the same length by cutting the longer one"""
    if len(p) < len(t):
        return p, t[0 : len(p)]
    elif len(t) < len(p):
        return p[0 : len(t)], t
    else:
        return p, t


def take_middle_third(signal: np.ndarray) -> np.ndarray:
    """return only the middle third of a signal"""
    third = len(signal) // 3
    return signal[third : len(signal) - third]


def change_duration(signal: np.ndarray, sampling_rate: int, new_duration: float) -> np.ndarray:
    """change the duration by taking a piece in the middle according to the new duration"""
    old_duration = len(signal) / sampling_rate
    fraction_to_take = new_duration / old_duration
    samples_to_take = int(fraction_to_take * len(signal))
    start_index = len(signal) // 2 - samples_to_take // 2
    return signal[start_index : start_index + samples_to_take]


def norm1d(signal: np.ndarray) -> np.ndarray:
    """normalize a 1d-signal to the range from 0 to 1"""
    return np.divide(signal, np.max(np.abs(signal), axis=0))


def norm2d(matrix: np.ndarray) -> np.ndarray:
    """normalize a each row in a matrix to the range from 0 to 1"""
    return np.apply_along_axis(norm1d, 0, matrix)


def add_phase_to_oscillator(
    s: np.ndarray, phase: float, period: float, sample_spacing: float, mode: str = "delay"
) -> np.ndarray:
    """
    add phase to an oscillator signal by rolling over the signal

    args:
        s: the oscillator signal
        phase: the phase to add in radians
        period: the period of the signal
        sample_spacing: the time between samples in the signal
        mode: the mode to use for rolling the signal
    """
    samples_per_period = period / sample_spacing
    normalized_phase = 0.5 * phase  # normalize phase from 0, 2 to [0, 1)
    samples_to_shift = int(samples_per_period * normalized_phase)
    if samples_to_shift == 0:
        return s
    if mode == "roll":
        return np.roll(s, samples_to_shift)
    return np.pad(s, (samples_to_shift, 0))[:-samples_to_shift]


def get_first_dx0_index(signal: np.ndarray) -> int:
    """get the index of the first value in a signal that has a derivative of 0"""
    d = np.diff(signal)
    return np.where(d <= 0)[0][0]


def remove_spice_startup(signal: np.ndarray) -> np.ndarray:
    """remove the spice startup transient from a signal"""
    return signal[get_first_dx0_index(signal) :]


def remove_spice_offset(signal: np.ndarray) -> np.ndarray:
    """remove the spice offset from a signal"""
    return signal - np.mean(signal)


def pad_spice_signal(signal: np.ndarray, length: int) -> np.ndarray:
    """mean pad the beginning of a spice signal to a given length"""
    to_pad = length - len(signal)
    return np.pad(signal, (to_pad, 0))


def clean_spice_signal(signal: np.ndarray, samples: int) -> np.ndarray:
    """remove spice startup transient and normalize to the range [0, 1]"""
    x = remove_spice_startup(signal)
    x = remove_spice_offset(x)
    return pad_spice_signal(x, samples)


def remove_offset_two_sided(signal: np.ndarray) -> np.ndarray:
    """remove the offset from a signal"""
    mean = np.mean(signal)
    if mean > 0:
        return signal - np.abs(mean)
    else:
        return signal + np.abs(mean)


def extract_single_period(
    signal: np.ndarray, sampling_rate: int, debug: bool = False
) -> Tuple[float, np.ndarray]:
    """extract a single period of a periodic signal"""
    peaks = find_peaks(signal)[0]
    if len(peaks) < 2:
        return None, None
    else:
        first_peak_pos, second_peak_pos = find_peaks(signal)[0][0:2]
    snippet = signal[first_peak_pos:second_peak_pos]
    snippet_length = second_peak_pos - first_peak_pos
    snippet_duration = snippet_length / sampling_rate
    # find minimum to roll the signal to the correct phase
    position_of_minimum = np.argmin(snippet)
    snippet = np.roll(snippet, -position_of_minimum)

    if debug:
        plt.plot(snippet)
        plt.title("snippet")
        plt.show()

    # center offset around zero
    snippet = snippet - np.mean(snippet)

    return snippet_duration, snippet


def extrapolate_from_period(
    single_period_signal: np.ndarray,
    sampling_rate: int,
    new_duration: float,
    phase_shift: Union[float, None] = None,
) -> np.ndarray:
    """extrapolate a signal from a single period to the desired duration

    args:
        single_period_signal: the signal of a single period
        sampling_rate: the sampling rate of the signal
        new_duration: the new duration of the signal in seconds
        phase_shift: the phase shift in radians, e.g. multiples of 2 yield the original signal
    """
    if phase_shift is not None:
        single_period_signal = np.roll(
            single_period_signal, int(phase_shift * len(single_period_signal) / 2)
        )

    period_duration = len(single_period_signal) / sampling_rate
    if new_duration < period_duration:
        fractional_duration = new_duration / period_duration
        fractional_samples = int(fractional_duration * len(single_period_signal))
        return single_period_signal[0:fractional_samples]

    full_periods = np.floor(new_duration // period_duration).astype(int)
    partial_period = new_duration % period_duration
    partial_period_samples = int(partial_period * sampling_rate)
    partial_period_snippet = single_period_signal[0:partial_period_samples]
    return np.concatenate([single_period_signal] * full_periods + [partial_period_snippet])


def extrapolate_oscillation(
    signal: np.ndarray,
    sampling_rate: int,
    new_duration: float,
    phase_shift: Union[float, None] = None,
) -> np.ndarray:
    """extrapolate an oscillation signal to a new duration

    args:
        signal: the signal to extrapolate, must be periodic
        sampling_rate: the sampling rate of the signal
        new_duration: the new duration of the signal in seconds
        phase_shift: the phase shift in periods, e.g. multiples of 2 yield the original signal

    returns:
        the extrapolated signal
    """
    peaks = find_peaks(signal)[0]
    if len(peaks) < 2:
        return None
    else:
        first_peak_pos, second_peak_pos = find_peaks(signal)[0][0:2]
    snippet = signal[first_peak_pos:second_peak_pos]
    snippet_length = second_peak_pos - first_peak_pos
    snippet_duration = snippet_length / sampling_rate
    # consider finding the minimum to roll the signal to the correct phase
    if phase_shift is not None and phase_shift != 0:
        snippet = np.roll(snippet, int(phase_shift * snippet_length / 2))

    # center offset around zero
    snippet = snippet - np.mean(snippet)

    full_periods = np.floor(new_duration // snippet_duration).astype(int)
    partial_period = new_duration % snippet_duration
    partial_period_samples = int(partial_period * sampling_rate)
    partial_period_snippet = snippet[0:partial_period_samples]
    return np.concatenate([snippet] * full_periods + [partial_period_snippet])
