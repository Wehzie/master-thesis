"""
Parameters for the Python signal generator.

Compared to params_python, the parameters in this file aim at low computational requirements.

Parameters are defined for both quantitative and qualitative experiments.

For each type of experiment three values are defined to generate a plot with a visible trend.
For example: 10, 20 and 30 oscillators.
"""

import gen_signal_args_types as party
import sweep_types as sweety
import const
import parameter_builder
import dist
import gen_signal_python
import shared_params_target

RNG = const.RNG
MAX_Z_OPS = 150
M_AVERAGES = 1
N_OSCILLATORS = 10
SAMPLES = 30
DURATION = 0.1  # s
SYNTH_FREQ = 1e3  # Hz

#### #### #### #### BASE PARAMETERS FOR SIGNAL GENERATION #### #### #### ####

py_rand_args_uniform = party.PythonSignalRandArgs(
    description="test base-parameters for drawing oscillators from a uniform distribution",
    n_osc=N_OSCILLATORS,
    duration=None,
    samples=SAMPLES,
    freq_dist=dist.Dist(RNG.uniform, low=1e5, high=1e6),
    amplitude=0.5,  # resembling 0.5 V amplitude of V02
    weight_dist=dist.WeightDist(
        RNG.uniform, low=0, high=10, n=N_OSCILLATORS
    ),  # scale down when <1 and scale up when >1
    phase_dist=dist.Dist(
        RNG.uniform, low=-1 / 3, high=1 / 3
    ),  # uniform 0 to 2 pi phase shift seems too wild
    offset_dist=dist.Dist(RNG.uniform, low=0, high=0),
    sampling_rate=11025,  # the sampling rate of the Magpie signal
)

py_rand_args_normal = party.PythonSignalRandArgs(
    description="test base-parameters for drawing from a normal distribution",
    n_osc=N_OSCILLATORS,
    duration=None,
    samples=SAMPLES,
    freq_dist=dist.Dist(RNG.normal, loc=5e5, scale=4e5),
    amplitude=0.5,
    weight_dist=dist.WeightDist(RNG.normal, loc=0.5, scale=0.5, n=N_OSCILLATORS),
    phase_dist=dist.Dist(RNG.normal, loc=0, scale=1 / 3),
    offset_dist=dist.Dist(RNG.normal, loc=0, scale=0),
    sampling_rate=11025,
)

#### #### #### #### BASE PARAMETERS MODIFIERS #### #### #### ####

test_targets = shared_params_target.build_test_targets(
    duration=DURATION, samples=SAMPLES, synth_freq=SYNTH_FREQ
)

target_sweep_samples = sweety.TargetSweep(
    "sample based targets before making the python signal generator adapt its sampling rate to the target",
    test_targets,
    py_rand_args_uniform,
    gen_signal_python.PythonSigGen(),
    max_z_ops=MAX_Z_OPS,
    m_averages=M_AVERAGES,
)

# TODO
# target_sweep_time = sweety.TargetSweep(
#     "time based targets",
#     sample_targets,
#     py_rand_args_uniform,
#     gen_signal_python.PythonSigGen(),
#     max_z_ops=MAX_Z_OPS,
#     m_averages=M_AVERAGES,
# )

n_osc_sweep = sweety.NOscSweep(
    n_osc=[20, 35, 50],
)

z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[0, 5e2, 1e3, 3e3],
)

num_samples_sweep = sweety.NumSamplesSweep([30, 60, 90])

duration_sweep = sweety.DurationSweep([0.001, 0.005, 0.01])


# sweep band from narrow to wide centering around 5e5, the median VO2 freq
freq_sweep_around_vo2 = sweety.FreqSweep(
    parameter_builder.append_normal(
        [
            dist.Dist(RNG.uniform, low=4.5e5, high=5.5e5),
            dist.Dist(RNG.uniform, low=4e5, high=6e5),
            dist.Dist(RNG.uniform, low=3e5, high=7e5),
        ]
    ),
)

# sweep band from narrow to wide while keeping the lower bound at 0
freq_sweep_from_zero = sweety.FreqSweep(
    parameter_builder.append_normal(
        [
            dist.Dist(RNG.uniform, low=1 / const.MAX_TARGET_DURATION, high=10 ** (p))
            for p in range(0, 3)
        ]
    ),
)


def init_weight_sweep(rand_args: party.PythonSignalRandArgs) -> sweety.WeightSweep:
    """init weight sweep with given rand_args"""
    return sweety.WeightSweep(
        parameter_builder.append_normal(
            [
                dist.WeightDist(RNG.uniform, low=0, high=1e1, n=N_OSCILLATORS),
                dist.WeightDist(RNG.uniform, low=0, high=1e2, n=N_OSCILLATORS),
                dist.WeightDist(RNG.uniform, low=0, high=1e3, n=N_OSCILLATORS),
            ]
        )
    )


weight_sweep = init_weight_sweep(py_rand_args_uniform)

# negative phase shift is unnecessary, as -1/2 pi is equivalent to 3/2 pi
phase_sweep = sweety.PhaseSweep(
    parameter_builder.append_normal(
        [
            dist.Dist(RNG.uniform, low=0, high=1 / 2),
            dist.Dist(RNG.uniform, low=0, high=1),
            dist.Dist(RNG.uniform, low=0, high=2),
        ]
    )
)

amplitude_sweep = sweety.AmplitudeSweep([0.5, 2, 5])

offset_sweep = sweety.OffsetSweep(
    parameter_builder.append_normal(
        [
            dist.Dist(
                RNG.uniform, low=0, high=0
            ),  # works better with experiment analysis than Dist(0)
            dist.Dist(RNG.uniform, low=-1, high=1),
            dist.Dist(RNG.uniform, low=-5, high=5),
        ]
    )
)
