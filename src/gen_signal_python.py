from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Tuple
from data_analysis import plot_pred_target, plot_signal
from data_io import load_data
from gen_signal_spice import load_sim_data

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt

from param_types import PythonSignalDetArgs, PythonSignalRandArgs

# the sampling rate of the target signal
DEFAULT_SAMPLING_RATE: Final = 11025
DEFAULT_AMPLITUDE: Final = 0.5 # resembling 0.5 V amplitude of V02

def gen_inv_sawtooth(
    duration: float,
    samples: int,
    freq: float,
    amplitude: int,
    weight: float,
    phase: float,
    offset_fctr: float,
    sampling_rate: int,
    visual: bool = False
    ) -> Tuple[range, np.ndarray]:
    """
    generate something between inverse sawtooth and triangle wave using scipy
    
    freq: frequency of the generated signal
    duration: length of the generated signal in seconds
    sampling_rate: number of samples per second
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
    offset = amplitude * weight * offset_fctr
    y = offset + weight * amplitude * signal.sawtooth(phase * np.pi + 2 * np.pi * freq * x_time, width=0.15)
    if visual:
        plot_signal(y, x_time, title="x-time")
        plot_signal(y, x_samples, title="x-samples")
    return x_samples, y

def sum_atomic_signals(args: PythonSignalRandArgs) -> Tuple[np.ndarray, List[PythonSignalDetArgs]]:
    """compose a signal of single oscillators"""
    signal_matrix = np.empty((args.n_osc, args.samples))
    det_arg_li = list()

    for i in range(args.n_osc):
        # determine a set of parameters for a single oscillator
        det_params = draw_params_random(args)
        # store the parameter set
        det_arg_li.append(det_params)
        # generate single oscillator signal and add to matrix
        single_signal = gen_inv_sawtooth(**det_params.__dict__)
        _, signal_matrix[i,:] = single_signal
    return signal_matrix, det_arg_li

def draw_params_random(args: PythonSignalRandArgs) -> PythonSignalDetArgs:
    """draw randomly from parameter pool"""
    # frequency
    f = np.random.uniform(args.f_lo, args.f_hi)
    
    duration = args.duration
    
    # samples
    samples = args.samples

    # weight
    # TODO: better control here, also check const behavior
    weight = np.random.uniform(0, 1)

    # random phase
    random_phase = args.random_phase

    return PythonSignalDetArgs(f, duration, samples, weight, random_phase)

def gen_custom_inv_sawtooth(freq: float):
    """a formula to compute the an inverse sawtooth"""
    x = np.linspace(1, 10, 100)
    a = 1 # amplitude
    p = 1 / freq # period
    y = (2*a) / np.pi * np.arctan(1 / np.tan(np.pi*x / p))
    plt.plot(x, y)
    plt.show()


def interpolate_signal():
    """load a ngspice generated oscillation and interpolate the signal"""
    df = load_sim_data(Path("data/example_single_oscillator/netlist0.cir.dat"))
    s = df.iloc[:,1] # column as series
    arr = s.to_numpy() # arr

    x = range(len(arr))
    f = interp1d(x, arr, kind="cubic") # interpolate
    plt.plot(x, arr, 'o', x, f(x), "-")
    plt.show()


def main():
    # generate a single signal from deterministic arguments
    args = PythonSignalDetArgs(duration=10, samples=None,
        freq=0.5,
        amplitude=1, weight=1,
        phase=0,
        offset_fctr=0,
        sampling_rate=DEFAULT_SAMPLING_RATE)

    gen_inv_sawtooth(**args.__dict__, visual=True)
    plt.show()

    exit()

    # generate a sum of signals from random variables
    args = PythonSignalRandArgs()
    atomic_signals, det_arg_li = sum_atomic_signals(args)
    sig_sum = sum(atomic_signals)
    plot_signal(sig_sum)

if __name__ == "__main__":
    main()


def test_sum_signals():
    """sum two signals"""
    s1 = gen_inv_sawtooth(1, 10)
    s2 = gen_inv_sawtooth(2, 10)
    s3 = (s1[0], s1[1] + s2[1])
    plt.plot(s1[0], s1[1], s2[0], s2[1], s3[0], s3[1])
    plt.show()

