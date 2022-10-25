"""parameters documented in code for reproducibility"""
from pathlib import Path
import numpy as np

from algo import SearchAlgo
from algo_las_vegas import LasVegas, LasVegasWeight
from algo_monte_carlo import MCExploit, MCOneShot, MCOneShotWeight
import param_types as party
import sweep_types as sweety
import data_preprocessor
import data_io
from typing import Final, List, Tuple
import const
rng = const.RNG

param_sweep_schedule = {
    "vo2_r1": {
        "changed_component": "R1",
        "dependent_component": "v(\"/A\")",
        
        "start": "5k",
        "stop": "150k",
        "step": "1k",

        "time_step": "5e-9",
        "time_stop": "10u",
        "time_start": "0",
    },
}

# parameters to generate bird sounds
bird_params = {
    "magpie_single_oscillator": {
        "trials": 1,

        "n_osc": 1,

        "v_in": 14,

        "r_last": 0,

        "r_control": 1e6,

        "r_min": 30e3,
        "r_max": 70e3,
        "r_dist": "uniform",

        "c_min": 300e-12,
        "c_max": 300e-12,
        "c_dist": "uniform",

        "time_step": 5e-9,
        "time_stop": 1e-5,
        "time_start": 0,

        "dependent_component": "v(osc1)",
    },
    "magpie_sum": {
        "trials": 1,

        "n_osc": 1,

        "v_in": 4,

        "r_last": 1, # only with sum architecture

        "r_control": 1e6,


        "r_min": 8e3,
        "r_max": 8e3,
        "r_dist": "uniform",

        "c_min": 40e-12,
        "c_max": 100e-12,
        "c_dist": "uniform",

        "time_step": 5e-9,
        "time_stop": "10e-6",
        "time_start": "0",

        "dependent_component": "v(sum)",
    },
    "magpie_tree": {
        "trials": 1,

        "branching": 2, # branching factor
        "height": 5, # tree height

        "v_in": 14,

        "r_tree": 0, # only with tree architecture

        "r_control": 1e6,

        "r_min": 30e3,
        "r_max": 70e3,
        "r_dist": "uniform",

        "c_min": 300e-12,
        "c_max": 300e-12,
        "c_dist": "uniform",

        "time_step": 5e-9,
        "time_stop": "1e-5",
        "time_start": "0",

        "dependent_component": "v(wire0)",
    }
}

def comp_loc(low: float, high: float) -> float:
    """compute loc (mean) for normal distribution from uniform distribution's low and high bounds"""
    return low + (high - low) / 2

def comp_scale(low: float, high: float) -> float:
    """compute scale (std dev) for normal distribution from a uniform distribution's low and high bounds"""
    return (high - low) / 2

def append_normal(uniform_li: List[party.Dist]) -> List[party.Dist]:
    """given a list of party.Dists, repeat the list with normal distributions"""
    DistType = type(uniform_li[0])
    norm_li = list()
    for d in uniform_li:
        if d.is_const():
            raise ValueError("experiments with constant distributions aren't supported, do low=const, high=const instead")
        if "n" in d.kwargs: # n is in kwargs only for WeightDist where we draw n
            n = d.kwargs["n"]
        else:
            n = None # can pass None for regular Dist type
        loc = comp_loc(d.kwargs["low"], d.kwargs["high"])
        scale = comp_scale(d.kwargs["low"], d.kwargs["high"])
        norm_li.append(DistType(rng.normal, loc=loc, scale=scale, n=n))
    return uniform_li + norm_li

py_rand_args_uniform = party.PythonSignalRandArgs(
    n_osc = 100,
    duration = None,
    samples = 300,
    freq_dist = party.Dist(rng.uniform, low=1e5, high=1e6),
    amplitude = 0.5,                                                    # resembling 0.5 V amplitude of V02
    weight_dist = party.WeightDist(rng.uniform, low=0, high=10, n=100),   # resistor doesn't amplify so not > 1
    phase_dist = party.Dist(rng.uniform, low=-1/3, high=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = party.Dist(rng.uniform, low=0, high=0),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

py_rand_args_normal = party.PythonSignalRandArgs(
    n_osc = 3000,
    duration = None,
    samples = 300,
    freq_dist = party.Dist(rng.normal, loc=5e5, scale=4e5),
    amplitude = 0.5,                                    # resembling 0.5 V amplitude of V02
    weight_dist = party.WeightDist(rng.normal, loc=0.5, scale=0.5, n=3000),   # resistor doesn't amplify so not > 1
    phase_dist = party.Dist(rng.normal, loc=0, scale=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = party.Dist(rng.normal, loc=0, scale=100/3),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

def init_freq_sweep() -> List[party.Dist]:
    freq_li = list()
    # sweep band from narrow to wide
    freq_li += append_normal([ 
            party.Dist(rng.uniform, low=1e5, high=1e6),
            party.Dist(rng.uniform, low=1e4, high=1e7),
            party.Dist(rng.uniform, low=1e3, high=1e8),
            party.Dist(rng.uniform, low=1e2, high=1e9),
            party.Dist(rng.uniform, low=1e1, high=1e10),
            party.Dist(rng.uniform, low=1e0, high=1e11),
        ])
    # sweep across narrow bands
    freq_li += append_normal(
        [party.Dist(rng.uniform, low=10**p, high=10**(p+1)) for p in range(0, 11)]
    )
    return freq_li

# def init_const_time_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.ConstTimeSweep:
#     return sweety.ConstTimeSweep(
#     freq_dist = init_freq_sweep(),
#     amplitude = [0.5, 5e0, 5e1, 5e2, 5e3, 5e4, 5e5, 5e6],
#     weight_dist = append_normal([
#         party.WeightDist(rng.uniform, low=0, high=1, n=rand_args.n_osc),
#         party.WeightDist(rng.uniform, low=0, high=1e1, n=rand_args.n_osc),
#         party.WeightDist(rng.uniform, low=0, high=1e2, n=rand_args.n_osc),
#         party.WeightDist(rng.uniform, low=0, high=1e3, n=rand_args.n_osc),
#         party.WeightDist(rng.uniform, low=0, high=1e4, n=rand_args.n_osc),
#         party.WeightDist(rng.uniform, low=0, high=1e5, n=rand_args.n_osc),
#     ]),
#     phase_dist = [party.Dist(0)] + append_normal([
#         party.Dist(rng.uniform, low=-1/5, high=1/5),
#         party.Dist(rng.uniform, low=-1/3, high=1/3),
#         party.Dist(rng.uniform, low=-1/2, high=1/2),
#         party.Dist(rng.uniform, low=-1, high=1),
#         party.Dist(rng.uniform, low=-2, high=2),
#         ])
#     )
# const_time_sweep = init_const_time_sweep(py_rand_args_uniform)

expo_time_sweep = sweety.ExpoTimeSweep(
    n_osc=[100, 200, 300, 500, 1000, 2000],
)

sampling_rate_sweep = sweety.SamplingRateSweep([0.01, 0.1, 0.5, 1])

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
    MCOneShotWeight,
    # LasVegas,
    # LasVegas,
    # MCExploit,
    # MCExploit,
]

def init_algo_args_for_sweep(rand_args: party.PythonSignalRandArgs,
target: np.ndarray,
max_z_ops: int) -> List[party.AlgoArgs]:
    return ([
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True),
    ])
    
def init_algo_sweep(target: np.ndarray) -> sweety.AlgoSweep:
    rand_args = py_rand_args_uniform
    algo_args = init_algo_args_for_sweep(rand_args, target, max_z_ops=5e3)
    return sweety.AlgoSweep(algo_list, algo_args, m_averages=20)

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
