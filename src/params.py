import numpy as np

from algo import SearchAlgo
from algo_las_vegas import LasVegas, LasVegasWeight
from algo_monte_carlo import MCExploit, MCOneShot
import param_types as party
import sweep_types as sweety
from typing import List
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

def init_py_timeless_sweep_args() -> List[party.Dist]:
    li = list()
    # compare uniform and normal distribution
    for d in [rng.uniform, rng.normal]:
        # start with VO2 freq band, then widen
        li += [
            party.Dist(d, low=1e5, high=1e6),
            party.Dist(d, low=1e4, high=1e7),
            party.Dist(d, low=1e3, high=1e8),
            party.Dist(d, low=1e2, high=1e9),
            party.Dist(d, low=1e1, high=1e10),
            party.Dist(d, low=1e0, high=1e11),
        ]
        # compare narrow bands
        for p in range(0, 11):
            li.append(party.Dist(d, low=10**p, high=10**(p+1)))
    return li

def append_normal(li: List) -> List: #List[party.Dist]
    """given a list of party.Dists, repeat the list with normal distributions"""
    try:
        return li + [party.Dist(rng.normal, d.kwargs["low"], d.kwargs["high"]) for d in li]
    except:
        print(len(li))
        print(li[0].kwargs)

py_rand_args_uniform = party.PythonSignalRandArgs(
    n_osc = 100,
    duration = None,
    samples = 300,
    f_dist = party.Dist(rng.uniform, low=1e5, high=1e6),
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
    f_dist = party.Dist(rng.normal, loc=5e5, scale=4e5),
    amplitude = 0.5,                                    # resembling 0.5 V amplitude of V02
    weight_dist = party.WeightDist(rng.normal, loc=0.5, scale=0.5, n=3000),   # resistor doesn't amplify so not > 1
    phase_dist = party.Dist(rng.normal, loc=0, scale=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = party.Dist(rng.normal, loc=0, scale=100/3),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

def init_const_time_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.ConstTimeSweep:
    return sweety.ConstTimeSweep(
    f_dist = init_py_timeless_sweep_args(),
    amplitude = [0.5, 5e0, 5e1, 5e2, 5e3, 5e4, 5e5, 5e6],
    weight_dist = append_normal([
        party.WeightDist(rng.uniform, low=0, high=1, n=rand_args.n_osc),
        party.WeightDist(rng.uniform, low=0, high=1e1, n=rand_args.n_osc),
        party.WeightDist(rng.uniform, low=0, high=1e2, n=rand_args.n_osc),
        party.WeightDist(rng.uniform, low=0, high=1e3, n=rand_args.n_osc),
        party.WeightDist(rng.uniform, low=0, high=1e4, n=rand_args.n_osc),
        party.WeightDist(rng.uniform, low=0, high=1e5, n=rand_args.n_osc),
    ]),
    phase_dist = [party.Dist(0)] + append_normal([
        party.Dist(rng.uniform, low=-1/5, high=1/5),
        party.Dist(rng.uniform, low=-1/3, high=1/3),
        party.Dist(rng.uniform, low=-1/2, high=1/2),
        party.Dist(rng.uniform, low=-1, high=1),
        party.Dist(rng.uniform, low=-2, high=2),
        ])
    )
cost_time_sweep = init_const_time_sweep(py_rand_args_uniform)

sweep_py_expo_time_args = sweety.ExpoTimeSweep(
    n_osc=[100, 200, 300, 500, 1000, 2000],
    sampling_rate_factor=[0.01, 0.1, 0.5, 1],
)

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
        party.AlgoArgs(rand_args, target, k_samples=10, weight_mode=False),
        party.AlgoArgs(rand_args, target, k_samples=100, weight_mode=True),
    ]
    return sweety.AlgoSweep(algo_list, algo_args, m_averages=2)
