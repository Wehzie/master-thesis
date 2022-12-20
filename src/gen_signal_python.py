from typing import List, Tuple, Union
import data_analysis

import numpy as np
import scipy.signal
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
            duration = samples/sampling_rate
            samples = int(samples) # **dataclass.__dict__ converts to float
        elif samples == None and duration != None:
            samples = int(duration * sampling_rate)
        elif samples != None and duration != None:
            assert samples == int(duration * sampling_rate)
        else:
            raise ValueError("either duration or number of samples must be specified")
        
        # ceil is used to handle float durations
        ceil_duration = np.ceil(duration)
        ceil_samples = int(ceil_duration * sampling_rate)
        
        time = np.linspace(0, ceil_duration, ceil_samples)[0:samples]
        signal = amplitude * scipy.signal.sawtooth(phase * np.pi + 2 * np.pi * freq * time, width=0.15)
        return signal, time

    @staticmethod
    def draw_single_oscillator(rand_args: party.PythonSignalRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Union[None, party.PythonSignalDetArgs]]:
        # determine a set of parameters for a single oscillator
        det_args = PythonSigGen.draw_params_random(rand_args)
        # generate single oscillator signal, discard x-range
        single_signal, time = PythonSigGen.gen_inv_sawtooth(**det_args.__dict__)
        if store_det_args:
            return single_signal, time, det_args
        return single_signal, time, None

    @staticmethod
    def draw_n_oscillators(rand_args: party.PythonSignalRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, List[Union[None, party.PythonSignalDetArgs]]]:
        signal_matrix = np.empty((rand_args.n_osc, rand_args.samples))
        det_arg_li = list()

        for i in range(rand_args.n_osc):
            single_signal, _, det_args = PythonSigGen.draw_single_oscillator(rand_args, store_det_args)
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
        
        sample_spacing = 1/10000
        print("gen_custom_inv_sawtooth")
        print(data_analysis.get_freq_from_fft_v2(y, sample_spacing))
        data_analysis.plot_signal(y, x, show=True)

    if True:
        # generate a single signal from deterministic arguments
        args = party.PythonSignalDetArgs(duration=10,
            samples=None,
            freq=0.5,
            amplitude=1,
            phase=2,
            sampling_rate=11025)

        signal, time = sig_generator.gen_inv_sawtooth(**args.__dict__)
        print("gen_inv_sawtooth")
        print(data_analysis.get_freq_from_fft_v2(signal, 1/11025))
        data_analysis.plot_signal(signal, time, show=True)

    # generate a sum of signals from random variables
    n_oscillators = 8
    f = 1
    sampling_rate = 20*f
    rand_args = party.PythonSignalRandArgs(
        n_osc = n_oscillators,
        duration = 10,
        samples = None,
        freq_dist = dist.Dist(const.RNG.uniform, low=1, high=50),
        #freq_dist = dist.Dist(f),
        amplitude = 0.5,
        weight_dist = dist.Dist(const.RNG.uniform, low=0.1, high=10, n=n_oscillators),
        #weight_dist=dist.Dist(1, n=n_oscillators),
        phase_dist = dist.Dist(const.RNG.uniform, low=0, high=2),
        #phase_dist = dist.Dist(0),
        offset_dist = dist.Dist(const.RNG.uniform, low=-1, high=1),
        #offset_dist = dist.Dist(0),
        sampling_rate = sampling_rate
    )
    if rand_args.samples == None:
        rand_args.samples = int(rand_args.duration * rand_args.sampling_rate)

    if True:
        print("single oscillator from rand_args")
        single_signal, time, _  = sig_generator.draw_single_oscillator(rand_args)
        print(data_analysis.get_freq_from_fft_v2(signal, 1/sampling_rate))
        data_analysis.plot_signal(single_signal, time)
        plt.show()

    time = rand_args.get_time()
    
    if True:
        print("sum of oscillators from rand_args")
        signal_matrix, _ = sig_generator.draw_n_oscillators(rand_args)
        sig_sum = np.sum(signal_matrix, axis=0)
        data_analysis.plot_signal(sig_sum, time)
        plt.show()

    if True:
        print("sample from rand_args")
        sample = sig_generator.draw_sample(rand_args)
        data_analysis.plot_signal(sample.weighted_sum, time)
        data_analysis.plot_individual_oscillators(sample.signal_matrix)
        #data_analysis.plot_f0_hist(sample.signal_matrix, 1/sampling_rate)
        plt.show()

if __name__ == "__main__":
    main()


