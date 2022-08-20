from pathlib import Path
from typing import Final
from sound_generator import load_sim_data

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# the sampling rate of the target signal
DEFAULT_SAMPLING_RATE: Final = 11025
DEFAULT_AMPLITUDE: Final = 0.5 # resembling 0.5 V amplitude of V02

def gen_sawtooth():
    """generate sawtooth using scipy"""
    x = np.linspace(0, 10, 500)
    plt.plot(x, signal.sawtooth(2 * np.pi * 5 * x))
    plt.show()

def gen_inv_sawtooth(
    freq: float,
    duration: float,
    amplitude: int = DEFAULT_AMPLITUDE, 
    weight: float = 1,
    random_phase: bool = False,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    visual: bool = False):
    """
    generate something between inverse sawtooth and triangle wave using scipy
    
    freq: frequency of the generated signal
    duration: length of the generated signal in seconds
    sampling_rate: number of samples per second
    """

    samples = duration * int(sampling_rate)
    
    phase = 0
    if random_phase:
        if freq > 0:
            phase = np.random.uniform(0, freq)
        else:
            phase = np.random.uniform(freq, 0)

    x = np.linspace(0, duration, samples)
    y = weight * amplitude * signal.sawtooth(2 * np.pi * freq * x + phase, width=0.15)
    if visual:
        plt.plot(x, y)
        plt.show()
    return x, y

def gen_inv_sawtooth_api(det_params: dict):
    """api to get_inv_sawtooth via det_params dict"""
    freq = det_params["f"]
    duration = det_params["duration"]
    random_phase = det_params["random_phase"]
    weight = det_params["weight"]
    return gen_inv_sawtooth(freq, duration, weight=weight, random_phase=random_phase)

def gen_custom_sawtooth():
    x = np.linspace(1, 10, 100)
    a = 1 # amplitude
    p = 2 # period
    y = - (2*a) / np.pi * np.arctan(1 / np.tan(np.pi*x / p))
    plt.plot(x, y)
    plt.show()

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
    gen_inv_sawtooth(freq=0.5, duration=10, weight=0.5, visual=True, sampling_rate=DEFAULT_SAMPLING_RATE)

if __name__ == "__main__":
    main()

def test_sum_signals():
    """sum two signals"""
    s1 = gen_inv_sawtooth(1, 10)
    s2 = gen_inv_sawtooth(2, 10)
    s3 = (s1[0], s1[1] + s2[1])
    plt.plot(s1[0], s1[1], s2[0], s2[1], s3[0], s3[1])
    plt.show()

