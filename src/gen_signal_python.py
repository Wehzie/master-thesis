from typing import List, Tuple, Union
import data_analysis

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import param_types as party
import const
import dist
import gen_signal

class PythonSigGen(gen_signal.SignalGenerator):

    @staticmethod
    def draw_params_random(args: party.PythonSignalRandArgs) -> party.PythonSignalDetArgs:
        duration = args.duration
        samples = args.samples
        freq = args.freq_dist.draw() # frequency
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
        visual: bool = False
        ) -> Tuple[range, np.ndarray]:
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
            duration = samples/sampling_rate
            samples = int(samples) # **dataclass.__dict__ converts to float
        elif samples == None and duration != None:
            samples = int(duration * sampling_rate)
        else:
            raise ValueError("either duration or number of samples must be specified")
        
        # ceil is used to handle float durations
        ceil_duration = np.ceil(duration)
        ceil_samples = int(ceil_duration * sampling_rate)
        
        x_time = np.linspace(0, ceil_duration, ceil_samples)[0:samples]
        x_samples = range(len(x_time))
        y = amplitude * signal.sawtooth(phase * np.pi + 2 * np.pi * freq * x_time, width=0.15)
        if visual:
            data_analysis.plot_signal(y, x_time, title="x-time")
            data_analysis.plot_signal(y, x_samples, title="x-samples")
        return x_samples, y

    @staticmethod
    def draw_single_oscillator(rand_args: party.PythonSignalRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, Union[None, party.PythonSignalDetArgs]]:
        # determine a set of parameters for a single oscillator
        det_args = PythonSigGen.draw_params_random(rand_args)
        # generate single oscillator signal, discard x-range
        _, single_signal = PythonSigGen.gen_inv_sawtooth(**det_args.__dict__)
        if store_det_args:
            return single_signal, det_args
        return single_signal, None

    @staticmethod
    def draw_n_oscillators(rand_args: party.PythonSignalRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, List[Union[None, party.PythonSignalDetArgs]]]:
        signal_matrix = np.empty((rand_args.n_osc, rand_args.samples))
        det_arg_li = list()

        for i in range(rand_args.n_osc):
            single_signal, det_args = PythonSigGen.draw_single_oscillator(rand_args, store_det_args)
            if store_det_args: det_arg_li.append(det_args)
            signal_matrix[i,:] = single_signal
        return signal_matrix, det_arg_li

def gen_custom_inv_sawtooth(
    duration: float,
    freq: float,
    amplitude: int,
    phase: float,
    offset: float,
    sampling_rate: int,
    ) -> np.ndarray:
    """formula to compute the inverse sawtooth without scipy"""
    x = np.linspace(1, duration, sampling_rate*duration)
    T = 1 / freq # period
    y = offset + (2*amplitude) / np.pi * np.arctan(1 / np.tan(np.pi*phase + np.pi*x / T))

    return x, y

def main():
    sig_generator = PythonSigGen()

    if True:
        x, y = gen_custom_inv_sawtooth(
            duration = 10,
            freq = 2,
            amplitude = 1,
            phase = 0,
            offset = 0,
            sampling_rate = 10000
        )
        data_analysis.plot_signal(y, x)
        plt.show()

        sample_spacing = 1/10000
        f0=data_analysis.get_freq_from_fft(y, sample_spacing)
        f1, _, _=data_analysis.get_freq_from_fft_v2(y, sample_spacing)
        print("fs:", f0, f1)

    if True:
        # generate a single signal from deterministic arguments
        args = party.PythonSignalDetArgs(duration=10, samples=None,
            freq=0.5,
            amplitude=1,
            phase=2,
            sampling_rate=11025)

        sig_generator.gen_inv_sawtooth(**args.__dict__, visual=True)
        plt.show()

    # generate a sum of signals from random variables
    rand_args = party.PythonSignalRandArgs(
        n_osc = 3,
        duration = None,
        samples = 300,
        freq_dist = dist.Dist(const.RNG.uniform, low=1e5, high=1e6),
        amplitude = 0.5,
        weight_dist = dist.Dist(const.RNG.uniform, low=0.1, high=1, n=3),
        phase_dist = dist.Dist(const.RNG.uniform, low=0, high=2),
        offset_dist = dist.Dist(const.RNG.uniform, low=-1/3, high=1/3),
        sampling_rate = 11025
    )

    if True:
        single_signal, y = sig_generator.draw_single_oscillator(rand_args)
        data_analysis.plot_signal(single_signal)
        plt.show()

    if True:
        signal_matrix, det_arg_li = sig_generator.draw_n_oscillators(rand_args)
        sig_sum = np.sum(signal_matrix, axis=0)
        data_analysis.plot_signal(sig_sum)
        plt.show()

    if True:
        sample = sig_generator.draw_sample(rand_args)
        data_analysis.plot_signal(sample.weighted_sum)
        plt.show()


if __name__ == "__main__":
    main()

