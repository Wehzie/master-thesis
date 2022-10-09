from pathlib import Path
from typing import List, Tuple
from data_analysis import  plot_signal
from data_io import load_sim_data

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt

import param_types as party
import params

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

def sum_atomic_signals(args: party.PythonSignalRandArgs, store_det_args: bool = False
) -> Tuple[np.ndarray, List[party.PythonSignalDetArgs]]:
    """compose a signal of single oscillators

    param:
        store_det_args: whether to store the deterministic parameters underlying each oscillator in a model
    """
    signal_matrix = np.empty((args.n_osc, args.samples))
    det_arg_li = list()

    for i in range(args.n_osc):
        # determine a set of parameters for a single oscillator
        det_params = draw_params_random(args)
        # store the parameter set
        if store_det_args: det_arg_li.append(det_params)
        # generate single oscillator signal and add to matrix
        single_signal = gen_inv_sawtooth(**det_params.__dict__)
        _, signal_matrix[i,:] = single_signal
    return signal_matrix, det_arg_li

def draw_params_random(args: party.PythonSignalRandArgs) -> party.PythonSignalDetArgs:
    """draw randomly from parameter pool"""
    duration = args.duration
    samples = args.samples
    freq = args.f_dist.draw() # frequency
    amplitude = args.amplitude
    weight = args.weight_dist.draw()    
    phase = args.phase_dist.draw()
    offset_fctr = args.offset_dist.draw()
    sampling_rate = args.sampling_rate

    return party.PythonSignalDetArgs(duration, samples, freq, amplitude, weight, phase, offset_fctr, sampling_rate)

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
    df = load_sim_data(Path("data/example_single_oscillator/netlist0.cir.dat"))
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

    # generate a single signal from deterministic arguments
    args = party.PythonSignalDetArgs(duration=10, samples=None,
        freq=0.5,
        amplitude=1, weight=1,
        phase=2,
        offset_fctr=-10,
        sampling_rate=11025)

    gen_inv_sawtooth(**args.__dict__, visual=True)
    plt.show()
    exit()

    # generate a sum of signals from random variables
    rng = np.random.default_rng(params.GLOBAL_SEED)
    args = PythonSignalRandArgs(
        n_osc = 3,
        duration = None,
        samples = 300,
        f_dist = Dist(rng.uniform, low=1e5, high=1e6),
        amplitude = 0.5,
        weight_dist = Dist(rng.uniform, low=0.1, high=1),
        phase_dist = Dist(rng.uniform, low=0, high=2),
        offset_dist = Dist(rng.uniform, low=-1/3, high=1/3),
        sampling_rate = 11025
    )
    atomic_signals, det_arg_li = sum_atomic_signals(args)
    sig_sum = sum(atomic_signals)
    plot_signal(sig_sum)
    plt.show()

if __name__ == "__main__":
    main()


def test_sum_signals():
    """sum two signals"""
    s1 = gen_inv_sawtooth(1, 10)
    s2 = gen_inv_sawtooth(2, 10)
    s3 = (s1[0], s1[1] + s2[1])
    plt.plot(s1[0], s1[1], s2[0], s2[1], s3[0], s3[1])
    plt.show()

