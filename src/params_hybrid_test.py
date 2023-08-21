"""This module defines production parameters for the SpiPy (Spice+Python, Hybrid) signal generator."""

import gen_signal_args_types as party
import dist
import const
import sweep_types as sweety
import shared_params_target
import gen_signal_spipy


RNG = const.RNG
MAX_Z_OPS = 150
M_AVERAGES = 1
N_OSCILLATORS = 10
SYNTH_FREQ = 5e5  # Hz

#### #### #### #### #### #### SIGNAL GENERATOR ARGUMENTS #### #### #### #### #### ####

# parameters for testing purposes
spice_single_det_args = party.SpiceSingleDetArgs(
    n_osc=1,
    v_in=14,
    r=140e3,
    r_last=1,
    r_control=1,
    c=300e-12,
    time_step=2e-9,
    time_stop=1e-4,
    time_start=0,
    dependent_component="v(osc1)",
    phase=0.5,
    generator_mode=party.SpipyGeneratorMode.CACHE,
    down_sample_factor=1 / 200,
)


spice_rand_args_uniform = party.SpiceSumRandArgs(
    description="VO2-RC circuit parameters inspired by Maffezzoni et al. 2015",
    n_osc=N_OSCILLATORS,
    v_in=14,
    r_last=1,
    r_control=1,
    r_dist=dist.Dist(RNG.uniform, low=20e3, high=140e3),
    c_dist=dist.Dist(300e-12),
    time_step=2e-9,
    time_stop=2e-5,
    time_start=0,
    dependent_component="v(osc1)",
    # Python controlled parameters
    phase_dist=dist.Dist(RNG.uniform, low=0, high=2),
    weight_dist=dist.WeightDist(RNG.uniform, low=0, high=10, n=N_OSCILLATORS),
    offset_dist=dist.Dist(RNG.uniform, low=-10, high=10),
    # runtime and memory optimizations
    generator_mode=party.SpipyGeneratorMode.CACHE,
    down_sample_factor=1 / 200,
)

#### #### #### #### #### #### EXPERIMENT PARAMETERS #### #### #### #### #### ####

target_sweep = sweety.TargetSweep(
    "evaluate the ability of the hybrid signal generator to fit a variety of targets",
    shared_params_target.build_test_targets(
        duration=spice_rand_args_uniform.get_duration(),
        sampling_rate=spice_rand_args_uniform.get_sampling_rate(),
        synth_freq=SYNTH_FREQ,
    ),
    spice_rand_args_uniform,
    gen_signal_spipy.SpipySignalGenerator(),
    max_z_ops=MAX_Z_OPS,
    m_averages=M_AVERAGES,
)

target_freq_sweep = sweety.TargetSweep(
    "evaluate the ability of the hybrid signal generator to fit varying frequency targets",
    shared_params_target.build_target_freq_sweep(
        duration=spice_rand_args_uniform.get_duration(),
        sampling_rate=spice_rand_args_uniform.get_sampling_rate(),
    ),
    spice_rand_args_uniform,
    gen_signal_spipy.SpipySignalGenerator(),
    max_z_ops=MAX_Z_OPS,
    m_averages=M_AVERAGES,
)

n_osc_sweep = sweety.NOscSweep(
    n_osc=[20, 30, 40],
)

z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[0, 100, 300, 500],
)

duration_sweep = sweety.DurationSweep([1e-4, 1e-3, 1e-2])

resistor_sweep = sweety.ResistorSweep(
    r_dist=[
        dist.Dist(RNG.uniform, low=40e3, high=54e3),
        dist.Dist(RNG.uniform, low=19e3, high=75e3),
        dist.Dist(RNG.uniform, low=19e3, high=125e3),
    ]
)

weight_sweep = sweety.WeightSweep(
    weight_dist=[
        dist.WeightDist(RNG.uniform, low=0, high=1e1, n=N_OSCILLATORS),
        dist.WeightDist(RNG.uniform, low=0, high=1e2, n=N_OSCILLATORS),
        dist.WeightDist(RNG.uniform, low=0, high=1e3, n=N_OSCILLATORS),
    ]
)

phase_sweep = sweety.PhaseSweep(
    phase_dist=[
        dist.Dist(RNG.uniform, low=0, high=0),
        dist.Dist(RNG.uniform, low=0, high=1),
        dist.Dist(RNG.uniform, low=0, high=2),
    ]
)

offset_sweep = sweety.OffsetSweep(
    offset_dist=[
        dist.Dist(RNG.uniform, low=0, high=0),  # works better with experiment analysis than Dist(0)
        dist.Dist(RNG.uniform, low=-1, high=1),
        dist.Dist(RNG.uniform, low=-5, high=5),
    ]
)
