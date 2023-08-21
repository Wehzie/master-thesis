"""Production parameters required by the Python signal generator."""

import gen_signal_args_types as party
import sweep_types as sweety
import const
import dist
import parameter_builder
import gen_signal_python
import shared_params_target

RNG = const.RNG
MAX_Z_OPS = 20000
M_AVERAGES = 10
N_OSCILLATORS = 100
SAMPLES = 300
DURATION = 0.1  # s
SYNTH_FREQ = 1e3  # Hz

#### #### #### #### BASE PARAMETERS FOR SIGNAL GENERATION #### #### #### ####

n_quality_test = 3000
py_rand_args_quality_test = party.PythonSignalRandArgs(
    description="for qualitative experiments; parameters for qualitative test of an algorithm; capable of fitting audio such that its recognized by human ear",
    n_osc=n_quality_test,
    duration=None,
    samples=15000,
    freq_dist=dist.Dist(RNG.uniform, low=1e5, high=1e6),
    amplitude=0.5,
    weight_dist=dist.WeightDist(RNG.uniform, low=0, high=1, n=n_quality_test),
    phase_dist=dist.Dist(RNG.uniform, low=-1 / 3, high=1 / 3),
    offset_dist=dist.Dist(RNG.uniform, low=0, high=0),
    sampling_rate=11025,
)

py_rand_args_uniform = party.PythonSignalRandArgs(
    description="for quantitative experiments; base-parameters for drawing oscillators from a uniform distribution",
    n_osc=N_OSCILLATORS,
    duration=None,
    samples=SAMPLES,
    freq_dist=dist.Dist(RNG.uniform, low=1e5, high=1e6),
    amplitude=0.5,
    weight_dist=dist.WeightDist(RNG.uniform, low=0, high=10, n=N_OSCILLATORS),
    phase_dist=dist.Dist(RNG.uniform, low=-1 / 3, high=1 / 3),
    offset_dist=dist.Dist(RNG.uniform, low=0, high=0),
    sampling_rate=11025,
)

#### #### BASE PARAMETERS MODIFIERS FOR QUANTITATIVE EXPERIMENTS #### ####

production_targets = shared_params_target.build_production_targets(
    duration=DURATION, samples=SAMPLES
)

target_sweep_samples = sweety.TargetSweep(
    "sample based targets before making the python signal generator adapt its sampling rate to the target",
    production_targets,
    py_rand_args_uniform,
    gen_signal_python.PythonSigGen(),
    max_z_ops=MAX_Z_OPS,
    m_averages=M_AVERAGES,
)

# TODO
# target_sweep_time = sweety.TargetSweep(
#     "time based targets",
#     shared_params_target.production_targets,
#     py_rand_args_uniform,
#     gen_signal_python.PythonSigGen(),
#     max_z_ops=MAX_Z_OPS,
#     m_averages=M_AVERAGES,
# )


n_osc_sweep = sweety.NOscSweep(
    n_osc=[50, 100, 250, 500, 1000],
)


z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[0, 5e2, 1e3, 5e3, 1e4, 1e4, 5e4],
)


num_samples_sweep = sweety.NumSamplesSweep([50, 100, 300, 500])

duration_sweep = sweety.DurationSweep([0.001, 0.01, 0.1, 0.5])


# sweep band from narrow to wide centering around 5e5, the median VO2 freq
freq_sweep_around_vo2 = sweety.FreqSweep(
    parameter_builder.append_normal(
        [
            dist.Dist(RNG.uniform, low=4.5e5, high=5.5e5),
            dist.Dist(RNG.uniform, low=4e5, high=6e5),
            dist.Dist(RNG.uniform, low=3e5, high=7e5),
            dist.Dist(RNG.uniform, low=1e5, high=1e6),  # the default
            dist.Dist(RNG.uniform, low=1e4, high=1e7),
        ]
    )
)


# sweep band from narrow to wide while keeping the lower bound at 0
freq_sweep_from_zero = sweety.FreqSweep(
    parameter_builder.append_normal(
        [
            dist.Dist(RNG.uniform, low=1 / const.MAX_TARGET_DURATION, high=10 ** (p))
            for p in range(0, 9)
        ]
    )
)


def init_weight_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.WeightSweep:
    """init weight sweep with given rand_args"""
    return sweety.WeightSweep(
        parameter_builder.append_normal(
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
phase_sweep = sweety.PhaseSweep(
    parameter_builder.append_normal(
        [
            dist.Dist(
                RNG.uniform, low=0, high=0
            ),  # works better with experiment analysis than Dist(0)
            dist.Dist(RNG.uniform, low=0, high=1 / 3),
            dist.Dist(RNG.uniform, low=0, high=1 / 2),
            dist.Dist(RNG.uniform, low=0, high=1),
            dist.Dist(RNG.uniform, low=0, high=2),
        ]
    )
)


amplitude_sweep = sweety.AmplitudeSweep([0.5, 5, 25, 50, 100])


offset_sweep = sweety.OffsetSweep(
    parameter_builder.append_normal(
        [
            dist.Dist(
                RNG.uniform, low=0, high=0
            ),  # works better with experiment analysis than Dist(0)
            dist.Dist(RNG.uniform, low=-1, high=1),
            dist.Dist(RNG.uniform, low=-5, high=5),
            dist.Dist(RNG.uniform, low=-25, high=25),
            dist.Dist(RNG.uniform, low=-50, high=50),
        ]
    )
)
