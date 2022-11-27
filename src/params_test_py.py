"""parameters for testing the pipeline with low computational requirements

use like:
import test_params as params
"""
import numpy as np

from algo import SearchAlgo
import algo_las_vegas as alave
import algo_monte_carlo as almoca
import algo_evolution as alevo
import algo_gradient as algra
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
    weight_dist = dist.WeightDist(rng.uniform, low=0, high=10, n=py_rand_args_n_osc),   # scale down when <1 and scale up when >1
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
        dist.Dist(rng.uniform, low=4.5e5, high=5.5e5),
        dist.Dist(rng.uniform, low=4e5, high=6e5),
        dist.Dist(rng.uniform, low=3e5, high=7e5),
    ]),
)

# sweep band from narrow to wide while keeping the lower bound at 0
freq_sweep_from_zero = sweety.FreqSweep(
    params.append_normal(
        [ party.Dist(rng.uniform, low=1/const.MAX_TARGET_DURATION, high=10**(p)) for p in range(0, 3) ]
    ),
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

# negative phase shift is unnecessary, as -1/2 pi is equivalent to 3/2 pi
phase_sweep = sweety.PhaseSweep(params.append_normal([
        dist.Dist(rng.uniform, low=0, high=1/3),
        dist.Dist(rng.uniform, low=0, high=1/2),
        dist.Dist(rng.uniform, low=0, high=1),
        dist.Dist(rng.uniform, low=0, high=2),
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

z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[0, 1e3, 5e3, 1e4],
)

sampling_rate_sweep = sweety.NumSamplesSweep([30, 60, 90])

def init_algo_sweep(target: np.ndarray, rand_args: party.PythonSignalRandArgs, max_z_ops: int = 3e3, m_averages: int = 1) -> sweety.AlgoSweep:
    """initialize a set of algorithms with varying arguments for a fixed target and a fixed set of rand_args from which to draw oscillators
    
    args:
        target: the target signal
        rand_args: the random variables from which oscillators are drawn
        z: the maximum number of z-operations to perform
        m_averages: the number of times to average the results of each algorithm"""
    one_shot_algos = [
        sweety.AlgoWithArgs(
            almoca.MCOneShot,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
        sweety.AlgoWithArgs(
            almoca.MCOneShotWeight,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
    ]
    exploit_algos = [
        sweety.AlgoWithArgs(
            almoca.MCExploit,
            party.AlgoArgs(rand_args, target, max_z_ops, j_replace=1),
        ),
        sweety.AlgoWithArgs(
            almoca.MCExploit,
            party.AlgoArgs(rand_args, target, max_z_ops, j_replace=10),
        ),
        sweety.AlgoWithArgs(
            almoca.MCExploitWeight,
            party.AlgoArgs(rand_args, target, max_z_ops, j_replace=1),
        ),
        sweety.AlgoWithArgs(
            almoca.MCExploitFast,
            party.AlgoArgs(rand_args, target, max_z_ops, j_replace=1),
        ),
    ]
    grow_shrink_algos = [
        sweety.AlgoWithArgs(
            almoca.MCGrowShrink,
            party.AlgoArgs(rand_args, target, max_z_ops, j_replace=1, l_damp_prob=0.5, h_damp_fac=0.5),
        ),
        sweety.AlgoWithArgs(
            almoca.MCDampen,
            party.AlgoArgs(rand_args, target, max_z_ops, j_replace=1, h_damp_fac=0.5),
        ),
        sweety.AlgoWithArgs(
            almoca.MCPurge,
            party.AlgoArgs(rand_args, target, max_z_ops, j_replace=1, h_damp_fac=0),
        ),
    ]
    anneal_algos = [
        sweety.AlgoWithArgs(
            almoca.MCAnneal,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
        sweety.AlgoWithArgs(
            almoca.MCAnnealWeight,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
        sweety.AlgoWithArgs(
            almoca.MCAnnealLog,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
        sweety.AlgoWithArgs(
            almoca.MCAnnealLogWeight,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
    ]
    las_vegas_algos = [
        sweety.AlgoWithArgs(
            alave.LasVegas,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
        sweety.AlgoWithArgs(
            alave.LasVegasWeight,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
    ]
    other_algos = [
        sweety.AlgoWithArgs(
            almoca.BasinHopping,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
        sweety.AlgoWithArgs(
            alevo.DifferentialEvolution,
            party.AlgoArgs(rand_args, target, max_z_ops),
        ),
    ]
    gradient_algos = [
            sweety.AlgoWithArgs(
            algra.LinearRegression,
            party.AlgoArgs(rand_args, target),
        ),
    ]

    all_algos_with_args = one_shot_algos + exploit_algos + grow_shrink_algos + anneal_algos + las_vegas_algos + other_algos


    
    return sweety.AlgoSweep(all_algos_with_args, m_averages)

# TODO: use for tests
algo_list: List[SearchAlgo] = [
    almoca.MCExploit,
    almoca.MCExploit,
    almoca.MCExploitWeight,
    almoca.MCExploitWeight,
    almoca.MCExploitFast,
    almoca.MCOneShot,
    almoca.MCOneShotWeight,
    almoca.MCGrowShrink,
    almoca.MCDampen,
    almoca.MCPurge,
    almoca.MCAnneal,
    almoca.MCAnnealWeight,
    almoca.MCAnnealLog,
    almoca.MCAnnealLogWeight,
    almoca.BasinHopping,
    alave.LasVegas,
    alave.LasVegasWeight,
    alevo.DifferentialEvolution,
    algra.LinearRegression,
]