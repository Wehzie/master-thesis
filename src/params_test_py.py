"""parameters for testing the pipeline with low computational requirements

use like:
import test_params as params
"""
import param_types as party
import sweep_types as sweety
import const
import param_util
import dist
rng = const.RNG

py_rand_args_n_osc = 10

py_rand_args_uniform = party.PythonSignalRandArgs(
    n_osc = py_rand_args_n_osc,
    duration = None,
    samples = 50,
    freq_dist = dist.Dist(rng.uniform, low=1e5, high=1e6),
    amplitude = 0.5,                                              # resembling 0.5 V amplitude of V02
    weight_dist = dist.WeightDist(rng.uniform, low=0, high=10, n=py_rand_args_n_osc),   # scale down when <1 and scale up when >1
    phase_dist = dist.Dist(rng.uniform, low=-1/3, high=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = dist.Dist(rng.uniform, low=0, high=0),
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

py_rand_args_normal = party.PythonSignalRandArgs(
    n_osc = py_rand_args_n_osc,
    duration = None,
    samples = 50,
    freq_dist = dist.Dist(rng.normal, loc=5e5, scale=4e5),
    amplitude = 0.5,                                    # resembling 0.5 V amplitude of V02
    weight_dist = dist.WeightDist(rng.normal, loc=0.5, scale=0.5, n=py_rand_args_n_osc),   # resistor doesn't amplify so not > 1
    phase_dist = dist.Dist(rng.normal, loc=0, scale=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = dist.Dist(rng.normal, loc=0, scale=0),    
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)

# sweep band from narrow to wide centering around 5e5, the median VO2 freq 
freq_sweep_around_vo2 = sweety.FreqSweep(
    param_util.append_normal([
        dist.Dist(rng.uniform, low=4.5e5, high=5.5e5),
        dist.Dist(rng.uniform, low=4e5, high=6e5),
        dist.Dist(rng.uniform, low=3e5, high=7e5),
    ]),
)

# sweep band from narrow to wide while keeping the lower bound at 0
freq_sweep_from_zero = sweety.FreqSweep(
    param_util.append_normal(
        [ dist.Dist(rng.uniform, low=1/const.MAX_TARGET_DURATION, high=10**(p)) for p in range(0, 3) ]
    ),
)

def init_weight_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.WeightSweep:
    """init weight sweep with given rand_args"""
    return sweety.WeightSweep(
        param_util.append_normal(
            [
            dist.WeightDist(rng.uniform, low=0, high=1e1, n=rand_args.n_osc),
            dist.WeightDist(rng.uniform, low=0, high=1e2, n=rand_args.n_osc),
            dist.WeightDist(rng.uniform, low=0, high=1e3, n=rand_args.n_osc),
            ]
        )
    )
weight_sweep = init_weight_sweep(py_rand_args_uniform)

# negative phase shift is unnecessary, as -1/2 pi is equivalent to 3/2 pi
phase_sweep = sweety.PhaseSweep(param_util.append_normal([
        dist.Dist(rng.uniform, low=0, high=1/2),
        dist.Dist(rng.uniform, low=0, high=1),
        dist.Dist(rng.uniform, low=0, high=2),
    ])
)

amplitude_sweep = sweety.AmplitudeSweep([0.5, 2, 5])

offset_sweep = sweety.OffsetSweep(
    param_util.append_normal([
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
