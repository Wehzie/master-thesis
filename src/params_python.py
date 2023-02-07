"""
Parameters for the Python signal generator.
Parameters are defined for both quantitative and qualitative experiments.

For each type of experiment three values are defined to generate a plot with a visible trend.
For example: 10, 20 and 30 oscillators.
"""

import param_types as party
import sweep_types as sweety
import const
import dist
import param_util
import meta_target
import gen_signal_python

RNG = const.RNG
MAX_Z_OPS = 10000
M_AVERAGES = 7
SAMPLES = 300
DURATION = 0.1 # s

#### #### #### #### BASE PARAMETERS FOR SIGNAL GENERATION #### #### #### ####

py_rand_args_n_osc = 3000
py_rand_args_quality_test = party.PythonSignalRandArgs(
    description = "for qualitative experiments; parameters for qualitative test of an algorithm; capable of fitting audio such that its recognized by human ear",
    n_osc=py_rand_args_n_osc,
    duration = None,
    samples = 15000,
    freq_dist = dist.Dist(RNG.uniform, low=1e5, high=1e6),
    amplitude = 0.5,
    weight_dist = dist.WeightDist(RNG.uniform, low=0, high=1, n=py_rand_args_n_osc),
    phase_dist = dist.Dist(RNG.uniform, low=-1/3, high=1/3),
    offset_dist = dist.Dist(RNG.uniform, low=0, high=0),
    sampling_rate = 11025,
)

py_rand_args_n_osc = 100
py_rand_args_uniform = party.PythonSignalRandArgs(
    description = "for quantitative experiments; base-parameters for drawing oscillators from a uniform distribution",
    n_osc = py_rand_args_n_osc,
    duration = None,
    samples = 300,
    freq_dist = dist.Dist(RNG.uniform, low=1e5, high=1e6),
    amplitude = 0.5,
    weight_dist = dist.WeightDist(RNG.uniform, low=0, high=10, n=py_rand_args_n_osc),
    phase_dist = dist.Dist(RNG.uniform, low=-1/3, high=1/3),
    offset_dist = dist.Dist(RNG.uniform, low=0, high=0),
    sampling_rate = 11025
)

py_rand_args_normal = party.PythonSignalRandArgs(
    description = "for quantitative experiments; base-parameters for drawing oscillators from a normal distribution",
    n_osc = py_rand_args_n_osc,
    duration = None,
    samples = 300,
    freq_dist = dist.Dist(RNG.normal, loc=5e5, scale=4e5),
    amplitude = 0.5,
    weight_dist = dist.WeightDist(RNG.normal, loc=5, scale=5, n=py_rand_args_n_osc),
    phase_dist = dist.Dist(RNG.normal, loc=1, scale=1),
    offset_dist = dist.Dist(RNG.normal, loc=0, scale=100/3),
    sampling_rate = 11025
)



#### #### BASE PARAMETERS MODIFIERS FOR QUANTITATIVE EXPERIMENTS #### #### 



# sweep band from narrow to wide centering around 5e5, the median VO2 freq 
freq_sweep_around_vo2 = sweety.FreqSweep(
    param_util.append_normal([
        dist.Dist(RNG.uniform, low=4.5e5, high=5.5e5),
        dist.Dist(RNG.uniform, low=4e5, high=6e5),
        dist.Dist(RNG.uniform, low=3e5, high=7e5),
        dist.Dist(RNG.uniform, low=1e5, high=1e6), # the default
        dist.Dist(RNG.uniform, low=1e4, high=1e7),
    ])
)


# sweep band from narrow to wide while keeping the lower bound at 0
freq_sweep_from_zero = sweety.FreqSweep(
    param_util.append_normal(
        [ dist.Dist(RNG.uniform, low=1/const.MAX_TARGET_DURATION, high=10**(p)) for p in range(0, 9) ]
    )
)


def init_weight_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.WeightSweep:
    """init weight sweep with given rand_args"""
    return sweety.WeightSweep(
        param_util.append_normal(
            [
            dist.WeightDist(RNG.uniform, low=0, high=1, n=rand_args.n_osc),
            dist.WeightDist(RNG.uniform, low=0, high=5, n=rand_args.n_osc),
            dist.WeightDist(RNG.uniform, low=0, high=10, n=rand_args.n_osc),
            dist.WeightDist(RNG.uniform, low=0, high=50, n=rand_args.n_osc),
            dist.WeightDist(RNG.uniform, low=0, high=100, n=rand_args.n_osc),
            ]
        )
    )
weight_sweep = init_weight_sweep(py_rand_args_uniform)


# phase shift is in radians
# negative phase shift is unnecessary, as -1/2 pi is equivalent to 3/2 pi
phase_sweep = sweety.PhaseSweep(param_util.append_normal([
        dist.Dist(RNG.uniform, low=0, high=0), # works better with experiment analysis than Dist(0)
        dist.Dist(RNG.uniform, low=0, high=1/3),
        dist.Dist(RNG.uniform, low=0, high=1/2),
        dist.Dist(RNG.uniform, low=0, high=1),
        dist.Dist(RNG.uniform, low=0, high=2),
    ])
)


amplitude_sweep = sweety.AmplitudeSweep([0.5, 2, 5, 10, 100])


offset_sweep = sweety.OffsetSweep(
    param_util.append_normal([
        dist.Dist(RNG.uniform, low=0, high=0), # works better with experiment analysis than Dist(0)
        dist.Dist(RNG.uniform, low=-1, high=1),
        dist.Dist(RNG.uniform, low=-5, high=5),
        dist.Dist(RNG.uniform, low=-25, high=25),
        dist.Dist(RNG.uniform, low=-50, high=50),
    ])
)


n_osc_sweep = sweety.NOscSweep(
    n_osc=[50, 100, 200, 500, 1000],
)


z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[0, 5e2, 1e3, 5e3, 1e4, 1e4, 5e4],
)


sampling_rate_sweep = sweety.NumSamplesSweep([50, 100, 300, 500])



sample_targets = [
    meta_target.SineTarget(DURATION, freq=50, samples=SAMPLES),
    meta_target.TriangleTarget(DURATION, freq=50, samples=SAMPLES),
    meta_target.SawtoothTarget(DURATION, freq=50, samples=SAMPLES),
    meta_target.InverseSawtoothTarget(DURATION, freq=50, samples=SAMPLES),
    meta_target.SquareTarget(DURATION, freq=50, samples=SAMPLES),
    meta_target.BeatTarget(DURATION, base_freq=50, samples=SAMPLES),
    meta_target.ChirpTarget(DURATION, start_freq=50, stop_freq=500, samples=SAMPLES),
    meta_target.DampChirpTarget(DURATION, start_freq=50, stop_freq=500, samples=SAMPLES),
    meta_target.SmoothGaussianNoiseTarget(DURATION, samples=SAMPLES),
    meta_target.SmoothUniformNoiseTarget(DURATION, samples=SAMPLES),
    meta_target.GaussianNoiseTarget(DURATION, samples=SAMPLES),
    meta_target.UniformNoiseTarget(DURATION, samples=SAMPLES),
]

python_target_sweep_sample = sweety.TargetSweep(
    "sample based targets before making the python signal generator adapt its sampling rate to the target",
    sample_targets,
    py_rand_args_uniform,
    gen_signal_python.PythonSigGen(),
    max_z_ops=MAX_Z_OPS,
    m_averages=M_AVERAGES,
)