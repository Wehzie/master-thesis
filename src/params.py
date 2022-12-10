"""parameters documented in code for reproducibility"""
import numpy as np

from algo import SearchAlgo
import algo_las_vegas as alave
import algo_monte_carlo as almoca
import algo_evolution as alevo
import param_types as party
import algo_args_types as algarty
import sweep_types as sweety
from typing import Final, List, Tuple
import const
import dist
import param_util
rng = const.RNG



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

# for fitting a sample that can be heared by human ear
py_rand_args_n_osc = 3000
py_rand_args_quality_test = party.PythonSignalRandArgs(
    n_osc=py_rand_args_n_osc,
    duration = None,
    samples = 15000,
    freq_dist = dist.Dist(rng.uniform, low=1e5, high=1e6),
    amplitude = 0.5,
    weight_dist = dist.WeightDist(rng.uniform, low=0, high=1, n=py_rand_args_n_osc),   # resistor doesn't amplify so not > 1
    phase_dist = dist.Dist(rng.uniform, low=-1/3, high=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = dist.Dist(rng.uniform, low=0, high=0),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025,
)

py_rand_args_n_osc = 100
py_rand_args_uniform = party.PythonSignalRandArgs(
    n_osc = py_rand_args_n_osc,
    duration = None,
    samples = 300,
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
    param_util.append_normal([
        dist.Dist(rng.uniform, low=1e5, high=1e6),
        dist.Dist(rng.uniform, low=1e4, high=1e7),
        dist.Dist(rng.uniform, low=1e3, high=1e8),
    ])
)

# TODO: sweep band from narrow to wide while keeping the lower bound at 0
# frequency 0 shouldn't be likely at all, start with 0.1 Hz
freq_sweep_from_zero = sweety.FreqSweep(
    param_util.append_normal(
        [ dist.Dist(rng.uniform, low=0, high=10**(p)) for p in range(0, 9) ]
    )
)

def init_weight_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.WeightSweep:
    """init weight sweep with given rand_args"""
    return sweety.WeightSweep(
        param_util.append_normal(
            [
            dist.WeightDist(rng.uniform, low=0, high=1e1, n=rand_args.n_osc),
            dist.WeightDist(rng.uniform, low=0, high=1e2, n=rand_args.n_osc),
            dist.WeightDist(rng.uniform, low=0, high=1e3, n=rand_args.n_osc),
            dist.WeightDist(rng.uniform, low=0, high=1e4, n=rand_args.n_osc),
            ]
        )
    )
weight_sweep = init_weight_sweep(py_rand_args_uniform)

phase_sweep = sweety.PhaseSweep(param_util.append_normal([
        dist.Dist(rng.uniform, low=-1/3, high=1/3),
        dist.Dist(rng.uniform, low=-1/2, high=1/2),
        dist.Dist(rng.uniform, low=-1, high=1),
        dist.Dist(rng.uniform, low=-2, high=2),
    ])
)

amplitude_sweep = sweety.AmplitudeSweep([0.5, 2, 5, 10])

offset_sweep = sweety.OffsetSweep(
    param_util.append_normal([
        dist.Dist(rng.uniform, low=0, high=0), # works better with experiment analysis than Dist(0)
        dist.Dist(rng.uniform, low=-1, high=1),
        dist.Dist(rng.uniform, low=-5, high=5),
        dist.Dist(rng.uniform, low=-25, high=25),
    ])
)

n_osc_sweep = sweety.NOscSweep(
    n_osc=[50, 100, 200, 500, 1000],
)

z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[5e2, 1e3, 5e3, 1e4, 5e4],
)

sampling_rate_sweep = sweety.NumSamplesSweep([50, 100, 200, 500])

# TODO: bundle args and algos
algo_list: List[SearchAlgo] = [
    # almoca.BasinHopping,
    alevo.DifferentialEvolution,
    alave.LasVegas,
    # alave.LasVegasWeight,
    # almoca.MCOneShot,
    # almoca.MCOneShotWeight,
    almoca.MCExploit,
    # almoca.MCExploit,
    # almoca.MCExploitWeight,
    # almoca.MCExploitWeight,
    # almoca.MCAnneal,
    # almoca.MCAnnealWeight,
    almoca.MCAnnealLog,
    # almoca.MCAnnealLogWeight,
    # almoca.MCPurge,
]

def init_algo_args_for_sweep(rand_args: party.PythonSignalRandArgs,
target: np.ndarray,
max_z_ops: int) -> List[algarty.AlgoArgs]:
    return ([                                                   # TODO: set weight mode and mp in constructor
    # basin hopping
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
    # differential evolution
        algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
    # las vegas
        algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, mp=const.MULTIPROCESSING),
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
    # mc one shot
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False),
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True),
    # mc exploit
        algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, j_replace=1, mp=const.MULTIPROCESSING),
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, j_replace=10, mp=const.MULTIPROCESSING),
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, j_replace=1, mp=const.MULTIPROCESSING),
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, j_replace=10, mp=const.MULTIPROCESSING),
    # mc anneal
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, mp=const.MULTIPROCESSING),
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
        algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=False, mp=const.MULTIPROCESSING),
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, mp=const.MULTIPROCESSING),
    # mc purge
        # algarty.AlgoArgs(rand_args, target, max_z_ops=max_z_ops, weight_mode=True, j_replace=1, mp=const.MULTIPROCESSING),
    ])
    
def init_algo_sweep(target: np.ndarray, rand_args: party.PythonSignalRandArgs) -> sweety.AlgoSweep:
    algo_args = init_algo_args_for_sweep(rand_args, target, max_z_ops=1e5)
    return sweety.AlgoSweep(algo_list, algo_args, m_averages=3)

