import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from gen_signal_python import *

def test_lazy():
    x, y = gen_custom_inv_sawtooth(
        duration = 10,
        freq = 1,
        amplitude = 1,
        phase = 0,
        offset = 0,
        sampling_rate = 10000
    )

    # generate a single signal from deterministic arguments
    args = party.PythonSignalDetArgs(duration=10, samples=None,
        freq=0.5,
        amplitude=1, weight=1,
        phase=2,
        offset_fctr=-10,
        sampling_rate=11025)
    gen_inv_sawtooth(**args.__dict__, visual=True)

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

    signal_matrix, det_arg_li = draw_n_signals(rand_args)
    sig_sum = np.sum(signal_matrix, axis=0)

    single_signal, y = draw_single_oscillator(rand_args)

    sample_x = draw_sample(rand_args)
