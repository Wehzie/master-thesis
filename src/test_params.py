"""parameters for testing the pipeline with low computational requirements

use like:
import test_params as params
"""
import numpy as np

from algo import SearchAlgo
import algo_las_vegas as alave
import algo_monte_carlo as almoca
import algo_evolution as alevo
import param_types as party
import sweep_types as sweety
from typing import Final, List, Tuple
import const
import params
import dist
rng = const.RNG

py_rand_args_n_osc = 40
py_rand_args_uniform = party.PythonSignalRandArgs(
    n_osc = py_rand_args_n_osc,
    duration = None,
    samples = 150,
    freq_dist = dist.Dist(rng.uniform, low=1e5, high=1e6),
    amplitude = 0.5,                                                    # resembling 0.5 V amplitude of V02
    weight_dist = dist.WeightDist(rng.uniform, low=0, high=10, n=py_rand_args_n_osc),   # resistor doesn't amplify so not > 1
    phase_dist = dist.Dist(rng.uniform, low=-1/3, high=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = dist.Dist(rng.uniform, low=0, high=0),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

py_rand_args_normal = party.PythonSignalRandArgs(
    n_osc = py_rand_args_n_osc,
    duration = None,
    samples = 300,
    freq_dist = dist.Dist(rng.normal, loc=5e5, scale=4e5),
    amplitude = 0.5,                                    # resembling 0.5 V amplitude of V02
    weight_dist = dist.WeightDist(rng.normal, loc=0.5, scale=0.5, n=py_rand_args_n_osc),   # resistor doesn't amplify so not > 1
    phase_dist = dist.Dist(rng.normal, loc=0, scale=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = dist.Dist(rng.normal, loc=0, scale=100/3),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

# sweep band from narrow to wide centering around 5e5, the median VO2 freq 
freq_sweep_around_vo2 = sweety.FreqSweep(
    params.append_normal([
        dist.Dist(rng.uniform, low=1e5, high=1e6),
        dist.Dist(rng.uniform, low=1e4, high=1e7),
        dist.Dist(rng.uniform, low=1e3, high=1e8),
    ]),
    #"freq_range_around_vo2"
)

# sweep band from narrow to wide while keeping the lower bound at 0
# frequency 0 shouldn't be likely at all, start with 0.1 Hz
freq_sweep_from_zero = sweety.FreqSweep(
    params.append_normal(
        [ party.Dist(rng.uniform, low=0, high=10**(p)) for p in range(0, 3) ]
    ),
    #"freq_range_from_zero"
)

def init_weight_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.WeightSweep:
    """init weight sweep with given rand_args"""
    return sweety.WeightSweep(
        params.append_normal(
            [
            dist.WeightDist(rng.uniform, low=0, high=1e1, n=rand_args.n_osc),
            dist.WeightDist(rng.uniform, low=0, high=1e2, n=rand_args.n_osc),
            dist.WeightDist(rng.uniform, low=0, high=1e3, n=rand_args.n_osc),
            ]
        )
    )
weight_sweep = init_weight_sweep(py_rand_args_uniform)

phase_sweep = sweety.PhaseSweep(params.append_normal([
        dist.Dist(rng.uniform, low=-1/3, high=1/3),
        dist.Dist(rng.uniform, low=-1/2, high=1/2),
        dist.Dist(rng.uniform, low=-1, high=1),
    ])
)

amplitude_sweep = sweety.AmplitudeSweep([0.5, 2, 5])

offset_sweep = sweety.OffsetSweep(
    params.append_normal([
        dist.Dist(rng.uniform, low=0, high=0), # works better with experiment analysis than Dist(0)
        dist.Dist(rng.uniform, low=-1, high=1),
        dist.Dist(rng.uniform, low=-5, high=5),
    ])
)

n_osc_sweep = sweety.NOscSweep(
    n_osc=[20, 35, 50],
)

# TODO: consider starting at lower number of operations
z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[0, 1e3, 5e3, 1e4],
)

sampling_rate_sweep = sweety.NumSamplesSweep([30, 60, 90])

algo_list: List[SearchAlgo] = [
    almoca.MCGrowShrink,
    almoca.MCDampen,
    almoca.MCExploitFast,
    almoca.BasinHopping,
    alevo.DifferentialEvolution,
    alave.LasVegas,
    alave.LasVegasWeight,
    almoca.MCOneShot,
    almoca.MCOneShotWeight,
    almoca.MCExploit,
    almoca.MCExploit,
    almoca.MCExploitWeight,
    almoca.MCExploitWeight,
    almoca.MCAnneal,
    almoca.MCAnnealWeight,
    almoca.MCAnnealLog,
    almoca.MCAnnealLogWeight,
    almoca.MCPurge,
]

def init_algo_args_for_sweep(rand_args: party.PythonSignalRandArgs,
target: np.ndarray,
max_z_ops: int) -> List[party.AlgoArgs]:
    return ([                                                   # TODO: set weight mode and mp in constructor
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, j_replace=1, l_damp_prob=0.5, h_damp_fac=0.5, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, j_replace=1, h_damp_fac=0.5, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, j_replace=1, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, j_replace=1, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, j_replace=10, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, j_replace=1, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, j_replace=10, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
        party.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, j_replace=1, h_damp_fac=0, mp=const.MULTIPROCESSING),
    ])
    
def init_algo_sweep(target: np.ndarray, rand_args: party.PythonSignalRandArgs) -> sweety.AlgoSweep:
    algo_args = init_algo_args_for_sweep(rand_args, target, max_z_ops=3e3)
    return sweety.AlgoSweep(algo_list, algo_args, m_averages=1)

