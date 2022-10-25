import os
import sys
from typing import Final, final
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from gen_signal_python import *
import data_preprocessor

def test_lazy():

    x, y = gen_custom_inv_sawtooth(
        duration = 10,
        freq = 1,
        amplitude = 1,
        phase = 0,
        offset = 0,
        sampling_rate = 10000
    )

    assert len(x) == 10000
    assert len(y) == 10000

    # generate a single signal from deterministic arguments
    args = party.PythonSignalDetArgs(duration=10, samples=None,
        freq=0.5,
        amplitude=1, 
        phase=2,
        sampling_rate=11025)

    x_samples, y = gen_inv_sawtooth(**args.__dict__, visual=False)

    assert len(x_samples) == 10*11025
    assert len(y) == 10*11025

    # generate a sum of signals from random variables
    rand_args = party.PythonSignalRandArgs(
        n_osc = 3,
        duration = None,
        samples = 300,
        freq_dist = party.Dist(const.TEST_RNG.uniform, low=1e5, high=1e6),
        amplitude = 0.5,
        weight_dist = party.WeightDist(const.TEST_RNG.uniform, low=0.1, high=1, n=3),
        phase_dist = party.Dist(const.TEST_RNG.uniform, low=0, high=2),
        offset_dist = party.Dist(const.TEST_RNG.uniform, low=-10, high=10),
        sampling_rate = 11025
    )

    _, target, _ = data_io.load_data(verbose=False)
    target = data_preprocessor.resample(target, rand_args.samples)
    assert len(target == rand_args.samples)

    single_oscillator, _ = draw_single_oscillator(rand_args)
    assert len(single_oscillator) == rand_args.samples

    signal_matrix, _ = draw_n_oscillators(rand_args)
    assert signal_matrix.shape == (rand_args.n_osc, rand_args.samples)

    base_sample: Final = draw_sample(rand_args, target)
    base_sample_copy = copy.deepcopy(base_sample) # mutable version of original object

    re_weighted_sample: Final = draw_sample_weights(base_sample_copy, rand_args, target)
    
    for s in [base_sample, re_weighted_sample]:
        assert s.signal_matrix.shape == (rand_args.n_osc, rand_args.samples)
        assert len(s.weights) == rand_args.n_osc
        assert isinstance(s.offset, int) or isinstance(s.offset, float)
        assert len(s.weighted_sum) == rand_args.samples
        assert isinstance(s.rmse, int) or isinstance(s.rmse, float)
