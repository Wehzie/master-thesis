"""parameters for testing the pipeline with low computational requirements

use like:
import test_params as params
"""
from pathlib import Path
import numpy as np

from algo import SearchAlgo
from algo_las_vegas import LasVegas, LasVegasWeight
from algo_monte_carlo import MCExploit, MCOneShot
import param_types as party
import sweep_types as sweety
import data_preprocessor
import data_io
from typing import Final, List, Tuple
import const
import params
rng = const.RNG


py_rand_args_uniform = party.PythonSignalRandArgs(
    n_osc = 30,
    duration = None,
    samples = 300,
    f_dist = party.Dist(rng.uniform, low=1e5, high=1e6),
    amplitude = 0.5,                                                    # resembling 0.5 V amplitude of V02
    weight_dist = party.WeightDist(rng.uniform, low=0, high=10, n=30),   # resistor doesn't amplify so not > 1
    phase_dist = party.Dist(rng.uniform, low=-1/3, high=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = party.Dist(rng.uniform, low=0, high=0),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

py_rand_args_normal = party.PythonSignalRandArgs(
    n_osc = 30,
    duration = None,
    samples = 300,
    f_dist = party.Dist(rng.normal, loc=5e5, scale=4e5),
    amplitude = 0.5,                                    # resembling 0.5 V amplitude of V02
    weight_dist = party.WeightDist(rng.normal, loc=0.5, scale=0.5, n=30),   # resistor doesn't amplify so not > 1
    phase_dist = party.Dist(rng.normal, loc=0, scale=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = party.Dist(rng.normal, loc=0, scale=100/3),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

def init_freq_sweep() -> List[party.Dist]:
    freq_li = list()
    # sweep band from narrow to wide
    freq_li += params.append_normal([ 
            party.Dist(rng.uniform, low=1e5, high=1e6),
            party.Dist(rng.uniform, low=1e4, high=1e7),
            party.Dist(rng.uniform, low=1e3, high=1e8),
        ])
    # sweep across narrow bands
    freq_li += params.append_normal(
        [party.Dist(rng.uniform, low=10**p, high=10**(p+1)) for p in range(0, 3)]
    )
    return freq_li

def init_const_time_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.ConstTimeSweep:
    return sweety.ConstTimeSweep(
    f_dist = init_freq_sweep(),
    amplitude = [0.5, 5e0, 5e1, 5e2],
    weight_dist = params.append_normal([
        party.WeightDist(rng.uniform, low=0, high=1e1, n=rand_args.n_osc),
        party.WeightDist(rng.uniform, low=0, high=1e2, n=rand_args.n_osc),
        party.WeightDist(rng.uniform, low=0, high=1e3, n=rand_args.n_osc),
    ]),
    phase_dist = [party.Dist(0)] + params.append_normal([
        party.Dist(rng.uniform, low=-1/3, high=1/3),
        party.Dist(rng.uniform, low=-1/2, high=1/2),
        party.Dist(rng.uniform, low=-1, high=1),
        ])
    )
const_time_sweep = init_const_time_sweep(py_rand_args_uniform)

expo_time_sweep = sweety.ExpoTimeSweep(
    n_osc=[50, 100],
)

sampling_rate_sweep = sweety.SamplingRateSweep([0.01, 0.1])

las_vegas_args = party.AlgoArgs(
    rand_args=py_rand_args_uniform,
    target=None,
    weight_mode=False,
    max_z_ops=None,
    k_samples=1,
    j_exploits=None,
    store_det_args=False,
    history=False,
    args_path=False,
)

algo_list: List[SearchAlgo] = [
    MCOneShot,
    MCOneShot, # weight mode
    # LasVegas,
    # LasVegas,
    # MCExploit,
    # MCExploit,
]

def init_algo_sweep(target: np.ndarray) -> sweety.AlgoSweep:
    rand_args = py_rand_args_uniform
    algo_args = [
        party.AlgoArgs(rand_args, target, k_samples=3, weight_mode=False),
        party.AlgoArgs(rand_args, target, k_samples=3, weight_mode=True),
    ]
    return sweety.AlgoSweep(algo_list, algo_args, m_averages=2)

def init_target2rand_args(scale_factor: float = 0.5) -> Tuple[party.PythonSignalRandArgs, Tuple]:
    """load, downsample target and inject number of samples into rand_args"""
    # loading and manipulating the target signal
    raw_sampling_rate, raw_target, raw_dtype = data_io.load_data()
    target_full_len: Final = data_preprocessor.sample_down_int(raw_target, scale_factor)
    # shorten the target
    target: Final = data_preprocessor.take_middle_third(target_full_len)
    # save to wav
    sampling_rate = int(scale_factor*raw_sampling_rate)
    data_io.save_signal_to_wav(target, sampling_rate, raw_dtype, Path("data/target_downsampled.wav"))
    # init search params
    rand_args = py_rand_args_uniform
    rand_args.samples = len(target) # generated signals match length of target
                                    # NOTE: the sampling rate could also be set lower instead
    return rand_args, (sampling_rate, target, raw_dtype)
