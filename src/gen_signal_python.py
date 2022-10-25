import copy
from pathlib import Path
from typing import List, Tuple, Union
import data_io
import data_analysis

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt

import param_types as party
import const
import sample

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


def draw_single_oscillator(rand_args: party.PythonSignalRandArgs, store_det_args: bool = False
) -> Tuple[np.ndarray, Union[None, party.PythonSignalDetArgs]]:
    """draw a single oscillator from random variables
    
    args:
        rand_args: a dataclass containing deterministic and random variables from which oscillators are drawn
        store_det_args: whether to store the deterministic parameters underlying each oscillator
    
    returns:
        single_signal: a single oscillator
        det_args: the deterministic parameters underlying the oscillator
    """
    # determine a set of parameters for a single oscillator
    det_args = draw_params_random(rand_args)
    # generate single oscillator signal, discard x-range
    _, single_signal = gen_inv_sawtooth(**det_args.__dict__)
    if store_det_args:
        return single_signal, det_args
    return single_signal, None


def draw_n_oscillators(rand_args: party.PythonSignalRandArgs, store_det_args: bool = False
) -> Tuple[np.ndarray, List[Union[None, party.PythonSignalDetArgs]]]:
    """compose a matrix of n-oscillators

    args:
        store_det_args: whether to store the deterministic parameters underlying each oscillator in a model

    returns:
        signal_matrix: a matrix of n-oscillators
        det_arg_li: a list of arguments used to generate each oscillator
    """
    signal_matrix = np.empty((rand_args.n_osc, rand_args.samples))
    det_arg_li = list()

    for i in range(rand_args.n_osc):
        single_signal, det_args = draw_single_oscillator(rand_args, store_det_args)
        if store_det_args: det_arg_li.append(det_args)
        signal_matrix[i,:] = single_signal
    return signal_matrix, det_arg_li


def draw_single_weight(rand_args: party.PythonSignalRandArgs) -> float:
    return rand_args.weight_dist.draw()


def draw_n_weights(rand_args: party.PythonSignalRandArgs) -> np.ndarray:
    return rand_args.weight_dist.draw_n()


def draw_offset(rand_args: party.PythonSignalRandArgs) -> float:
    return rand_args.offset_dist.draw()


def draw_params_random(args: party.PythonSignalRandArgs) -> party.PythonSignalDetArgs:
    """draw randomly from parameter pool"""
    duration = args.duration
    samples = args.samples
    freq = args.freq_dist.draw() # frequency
    amplitude = args.amplitude
    phase = args.phase_dist.draw()
    sampling_rate = args.sampling_rate

    return party.PythonSignalDetArgs(duration, samples, freq, amplitude, phase, sampling_rate)


def draw_sample(rand_args: party.PythonSignalRandArgs, target: Union[None, np.ndarray] = None,
store_det_args: bool = False) -> sample.Sample:
    """draw a sample from scratch and compute available metrics
    
    args:
        rand_args: a dataclass containing deterministic and random variables from which oscillators are drawn
        target: a target signal to compare the generated signal to
        store_det_args: whether to store the deterministic parameters underlying each oscillator in a model

    returns:
        a sample containing the generated signal
        sample contains matrix, weights, weighted sum, offset, rmse and deterministic arguments underlying the oscillators in the model
    """
    signal_matrix, det_args = draw_n_oscillators(rand_args, store_det_args)
    weights = draw_n_weights(rand_args)
    offset = draw_offset(rand_args)
    weighted_sum = sample.Sample.compute_weighted_sum(signal_matrix, weights, offset)
    rmse = None
    if target is not None:
        rmse = data_analysis.compute_rmse(weighted_sum, target)
    return sample.Sample(signal_matrix, weights, weighted_sum, offset, rmse, det_args)


def draw_sample_weights(base_sample: sample.Sample, rand_args: party.PythonSignalRandArgs, target: Union[None, np.ndarray] = None) -> sample.Sample:
    """return the base sample with different weights and recomputed metrics
    
    args:
        base_sample: the sample to copy
        rand_args: a dataclass containing deterministic and random variables from which the weights are drawn
    
    returns:
        the base sample with new weights and re-computed metrics
    """
    updated_sample = copy.deepcopy(base_sample)
    updated_sample.weights = draw_n_weights(rand_args)
    # TODO: should I draw a new offset?
    # updated_sample.offset = draw_offset(rand_args)
    updated_sample.weighted_sum = sample.Sample.compute_weighted_sum(updated_sample.signal_matrix, updated_sample.weights, updated_sample.offset)
    updated_sample.rmse = None
    if target is not None:
        updated_sample.rmse = data_analysis.compute_rmse(updated_sample.weighted_sum, target)   
    return updated_sample


def gen_custom_inv_sawtooth(
    duration: float,
    freq: float,
    amplitude: int,
    phase: float,
    offset: float,
    sampling_rate: int,
    ) -> np.ndarray:
    """a formula to compute the an inverse sawtooth"""
    x = np.linspace(1, duration, sampling_rate)
    T = 1 / freq # period
    y = offset + (2*amplitude) / np.pi * np.arctan(1 / np.tan(np.pi*phase + np.pi*x / T))

    return x, y


def interpolate_signal():
    """load a ngspice generated oscillation and interpolate the signal"""
    df = data_io.load_sim_data(Path("data/example_single_oscillator/netlist0.cir.dat"))
    s = df.iloc[:,1] # column as series
    arr = s.to_numpy() # arr

    x = range(len(arr))
    f = interp1d(x, arr, kind="cubic") # interpolate
    plt.plot(x, arr, 'o', x, f(x), "-")
    plt.show()


def main():
    if False:
        x, y = gen_custom_inv_sawtooth(
            duration = 10,
            freq = 1,
            amplitude = 1,
            phase = 0,
            offset = 0,
            sampling_rate = 10000
        )
        plot_signal(y)
        plt.show()

    if False:
        # generate a single signal from deterministic arguments
        args = party.PythonSignalDetArgs(duration=10, samples=None,
            freq=0.5,
            amplitude=1, weight=1,
            phase=2,
            offset_fctr=-10,
            sampling_rate=11025)

        gen_inv_sawtooth(**args.__dict__, visual=True)
        plt.show()

    # generate a sum of signals from random variables
    rand_args = party.PythonSignalRandArgs(
        n_osc = 3,
        duration = None,
        samples = 300,
        freq_dist = party.Dist(const.RNG.uniform, low=1e5, high=1e6),
        amplitude = 0.5,
        weight_dist = party.Dist(const.RNG.uniform, low=0.1, high=1, n=3),
        phase_dist = party.Dist(const.RNG.uniform, low=0, high=2),
        offset_dist = party.Dist(const.RNG.uniform, low=-1/3, high=1/3),
        sampling_rate = 11025
    )

    if False:
        signal_matrix, det_arg_li = draw_n_oscillators(rand_args)
        sig_sum = np.sum(signal_matrix, axis=0)
        data_analysis.plot_signal(sig_sum)
        plt.show()

    if False:
        single_signal, y = draw_single_oscillator(rand_args)
        data_analysis.plot_signal(single_signal)
        plt.show()

    if False:
        sample = draw_sample(rand_args)
        data_analysis.plot_signal(sample.weighted_sum)
        plt.show()


if __name__ == "__main__":
    main()

