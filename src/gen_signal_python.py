"""
This module implements the Python signal generator class.

Signals are generated fully within the Python environment.
"""

from typing import List, Tuple, Union
import data_analysis

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import gen_signal_args_types as party
import const
import dist
import gen_signal


class PythonSigGen(gen_signal.SignalGenerator):
    """Python signal generator class."""

    @staticmethod
    def draw_params_random(args: party.PythonSignalRandArgs) -> party.PythonSignalDetArgs:
        """draw parameters from the provided distributions"""
        duration = args.duration
        samples = args.samples
        freq = args.freq_dist.draw()  # frequency
        amplitude = args.amplitude
        phase = args.phase_dist.draw()
        sampling_rate = args.sampling_rate

        return party.PythonSignalDetArgs(duration, samples, freq, amplitude, phase, sampling_rate)

    @staticmethod
    def gen_inv_sawtooth(
        duration: float,
        samples: int,
        freq: float,
        amplitude: int,
        phase: float,
        sampling_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """generate a signal between inverse sawtooth and triangle wave using scipy

        args:
            freq: frequency of the generated signal
            duration: length of the generated signal in seconds
            sampling_rate: number of samples per second

        returns:
            x_samples: a range of integers incrementing by 1 from 0 to the number of samples of the signal
            y: a numpy array of the generated signal
        """
        if duration == None and samples != None:
            duration = samples / sampling_rate
            samples = int(samples)  # **dataclass.__dict__ converts to float
        elif samples == None and duration != None:
            samples = int(duration * sampling_rate)
        elif samples != None and duration != None:
            assert samples == int(duration * sampling_rate)
        else:
            raise ValueError("either duration or number of samples must be specified")

        # ceil is used to handle float durations
        ceil_duration = np.ceil(duration)
        ceil_samples = int(ceil_duration * sampling_rate)

        # NOTE: endpoint=True might yield lower RMSE, but also samples unevenly for fixed frequency signals
        time = np.linspace(0, ceil_duration, ceil_samples, endpoint=False)[0:samples]
        signal = amplitude * scipy.signal.sawtooth(
            phase * np.pi + 2 * np.pi * freq * time, width=0.15
        )
        return signal, time

    @staticmethod
    def draw_single_oscillator(
        rand_args: party.PythonSignalRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Union[None, party.PythonSignalDetArgs]]:
        """generate a time-series for a single oscillator, with a random frequency and phase"""
        # determine a set of parameters for a single oscillator
        det_args = PythonSigGen.draw_params_random(rand_args)
        # generate single oscillator signal, discard x-range
        single_signal, time = PythonSigGen.gen_inv_sawtooth(**det_args.__dict__)
        if store_det_args:
            return single_signal, time, det_args
        return single_signal, time, None

    @staticmethod
    def draw_n_oscillators(
        rand_args: party.PythonSignalRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, List[Union[None, party.PythonSignalDetArgs]]]:
        """draw a matrix of n oscillator signals, each with a random frequency and phase"""
        signal_matrix = np.empty((rand_args.n_osc, rand_args.samples))
        det_arg_li = list()

        for i in range(rand_args.n_osc):
            single_signal, _, det_args = PythonSigGen.draw_single_oscillator(
                rand_args, store_det_args
            )
            if store_det_args:
                det_arg_li.append(det_args)
            signal_matrix[i, :] = single_signal
        return signal_matrix, det_arg_li


def gen_custom_inv_sawtooth(
    duration: float,
    freq: float,
    amplitude: int,
    phase: float,
    offset: float,
    sampling_rate: int,
) -> np.ndarray:
    """compute the inverse sawtooth without scipy"""
    x = np.linspace(1, duration, sampling_rate * duration, endpoint=False)
    T = 1 / freq  # period
    y = offset + (2 * amplitude) / np.pi * np.arctan(1 / np.tan(np.pi * phase + np.pi * x / T))

    return x, y


def skewed_triangle(
    x: float, L: float, m: int, amplitude: float, phase: float, offset: float
) -> float:
    """
    return the value of a (skewed) triangle wave at point x

    args:
        x: point in time, in [0, 2*L]
        L: period length of the triangle wave
        m: skewness, m=1 is a triangle wave, as m grows a sawtooth is approximated
        amplitude: amplitude of the triangle wave
        phase: phase of the triangle wave
        offset: offset in radians

    references:
        - https://mathworld.wolfram.com/FourierSeriesTriangleWave.html
    """
    assert 0 <= x <= 2 * L, "x must be in [0, 2*L]"
    zero_bias = -amplitude / 2
    offset += zero_bias
    x = (x + L * phase / 2) % L

    def cond1(x, L, m):
        return (m * x * 1 / L) * amplitude + offset

    def cond2(x, L, m):
        return (1 - (m * (x - (L / m))) / ((m - 1) * L)) * amplitude + offset

    def cond3(x, L, m):
        return ((m * (x - 2 * L)) / L) * amplitude + offset

    if 0 <= x <= L / m:
        return cond1(x, L, m)
    elif L / m <= x <= 2 * L - L / m:
        return cond2(x, L, m)
    elif 2 * L - L / m <= x <= 2 * L:
        return cond3(x, L, m)
    else:
        raise ValueError("x must be in [0, 2*L]")


def skewed_triangle_wave(
    duration: float, sampling_rate: int, L: float, m: int, a: float, p: float, b: float
) -> np.ndarray:
    """
    return a (skewed) triangle wave of a given duration and sampling rate

    args:
        duration: in seconds
        sampling_rate: in Hz
        L: period length
        m: skewness, m=1 is a triangle wave, as m grows a sawtooth is approximated
        a: amplitude
        p: phase in radians
        b: offset

    # TODO: transition between cycles is not smooth
    # TODO: optimize with map or np.vectorize
    # TODO: phase, offset and frequency as optional argument instead of period
    """
    n_full_cycles = np.floor(duration / L).astype(int)
    n_samples_last_cycle = int((duration / L % 1) * sampling_rate)
    samples_per_period = int(sampling_rate / L)
    x_full = np.linspace(0, L * 2, int(samples_per_period * 2))
    y_full = [skewed_triangle(x_i, L, m, a, p, b) for x_i in x_full] * n_full_cycles
    x_last = np.linspace(0, L * 2, n_samples_last_cycle)
    y_rest = [skewed_triangle(x_i, L, m, a, p, b) for x_i in x_last]
    return np.array(y_full + y_rest)


def main():
    sig_generator = PythonSigGen()

    time = np.linspace(0, 1, 80000)
    fig = plt.figure()
    plt.plot(time, skewed_triangle_wave(duration=1, sampling_rate=10000, L=0.5, m=8, a=1, p=0, b=0))
    # x label
    fig.set_size_inches(10, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("Time [s]")
    plt.savefig("skewed_triangle.png", dpi=300)
    plt.show()
    exit()

    if False:
        x, y = gen_custom_inv_sawtooth(
            duration=10, freq=2, amplitude=1, phase=5, offset=0, sampling_rate=10000
        )

        sample_spacing = 1 / 10000
        print("gen_custom_inv_sawtooth")
        print(data_analysis.get_freq_from_fft_v2(y, sample_spacing))
        data_analysis.plot_signal(y, x, show=True)

    if False:
        # generate a single signal from deterministic arguments
        args = party.PythonSignalDetArgs(
            duration=10, samples=None, freq=0.5, amplitude=1, phase=2, sampling_rate=11025
        )

        signal, time = sig_generator.gen_inv_sawtooth(**args.__dict__)
        print("gen_inv_sawtooth")
        print(data_analysis.get_freq_from_fft_v2(signal, 1 / 11025))
        data_analysis.plot_signal(signal, time, show=True)

    # generate a sum of signals from random variables
    n_oscillators = 4
    f_max = 5
    sampling_rate = 2 * const.OVERSAMPLING_FACTOR * f_max
    rand_args = party.PythonSignalRandArgs(
        description="test Python signal generator",
        n_osc=n_oscillators,
        duration=10,
        samples=None,
        freq_dist=dist.Dist(const.RNG.uniform, low=1, high=f_max),
        # freq_dist = dist.Dist(f),
        amplitude=0.5,
        weight_dist=dist.Dist(const.RNG.uniform, low=0.1, high=10, n=n_oscillators),
        # weight_dist=dist.Dist(1, n=n_oscillators),
        phase_dist=dist.Dist(const.RNG.uniform, low=0, high=2),
        # phase_dist = dist.Dist(0),
        offset_dist=dist.Dist(const.RNG.uniform, low=-1, high=1),
        # offset_dist = dist.Dist(0),
        sampling_rate=sampling_rate,
    )
    if rand_args.samples == None:
        rand_args.samples = int(rand_args.duration * rand_args.sampling_rate)

    if False:
        print("single oscillator from rand_args")
        single_signal, time, _ = sig_generator.draw_single_oscillator(rand_args)
        print(data_analysis.get_freq_from_fft_v2(signal, 1 / sampling_rate))
        data_analysis.plot_signal(single_signal, time)
        plt.show()

    time = rand_args.get_time()

    if True:
        print("sum of oscillators from rand_args")
        signal_matrix, _ = sig_generator.draw_n_oscillators(rand_args)
        sig_sum = np.sum(signal_matrix, axis=0)
        from pathlib import Path

        data_analysis.plot_signal(
            sig_sum, time, rand_args.sampling_rate, save_path=Path("python_sum_combined.png")
        )
        data_analysis.plot_individual_oscillators(
            signal_matrix, time, save_path=Path("python_sum_individual.png")
        )
        plt.show()

    if False:
        print("sample from rand_args")
        sample = sig_generator.draw_sample(rand_args)
        data_analysis.plot_signal(sample.weighted_sum, time)
        data_analysis.plot_individual_oscillators(sample.signal_matrix, time)
        # data_analysis.plot_f0_hist(sample.signal_matrix, 1/sampling_rate)
        plt.show()


if __name__ == "__main__":
    main()
