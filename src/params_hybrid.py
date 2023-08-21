"""This module defines production parameters for the SpiPy (Spice+Python, Hybrid) signal generator."""

import gen_signal_args_types as party
import dist
import const
import sweep_types as sweety
import shared_params_target
import gen_signal_spipy

RNG = const.RNG
MAX_Z_OPS = 20000
M_AVERAGES = 10
N_OSCILLATORS = 100
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

production_targets = shared_params_target.build_production_targets(
    duration=spice_rand_args_uniform.get_duration(),
    sampling_rate=spice_rand_args_uniform.get_sampling_rate(),
)

target_sweep = sweety.TargetSweep(
    "evaluate the ability of the hybrid signal generator to fit a variety of targets",
    production_targets,
    spice_rand_args_uniform,
    gen_signal_spipy.SpipySignalGenerator(),
    max_z_ops=MAX_Z_OPS,
    m_averages=M_AVERAGES,
)

target_freq_sweep = sweety.TargetSweep(
    "evaluate the ability of the hybrid signal generator to fit varying frequency sinusoids",
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
    n_osc=[50, 100, 200, 500, 1000],
)

z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[0, 5e2, 1e3, 5e3, 1e4, 5e4],
)

duration_sweep = sweety.DurationSweep([1e-4, 1e-3, 1e-2])

resistor_sweep = sweety.ResistorSweep(
    r_dist=[
        dist.Dist(RNG.uniform, low=40e3, high=54e3),
        dist.Dist(RNG.uniform, low=33e3, high=61e3),
        dist.Dist(RNG.uniform, low=26e3, high=68e3),
        dist.Dist(RNG.uniform, low=19e3, high=75e3),
        dist.Dist(RNG.uniform, low=19e3, high=95e3),
        dist.Dist(RNG.uniform, low=19e3, high=125e3),
    ]
)

weight_sweep = sweety.WeightSweep(
    weight_dist=[
        dist.WeightDist(RNG.uniform, low=0, high=1, n=N_OSCILLATORS),
        dist.WeightDist(RNG.uniform, low=0, high=5, n=N_OSCILLATORS),
        dist.WeightDist(RNG.uniform, low=0, high=10, n=N_OSCILLATORS),
        dist.WeightDist(RNG.uniform, low=0, high=50, n=N_OSCILLATORS),
        dist.WeightDist(RNG.uniform, low=0, high=100, n=N_OSCILLATORS),
    ]
)

phase_sweep = sweety.PhaseSweep(
    phase_dist=[
        dist.Dist(RNG.uniform, low=0, high=0),  # works better with experiment analysis than Dist(0)
        dist.Dist(RNG.uniform, low=0, high=1 / 3),
        dist.Dist(RNG.uniform, low=0, high=1 / 2),
        dist.Dist(RNG.uniform, low=0, high=1),
        dist.Dist(RNG.uniform, low=0, high=2),
    ]
)

offset_sweep = sweety.OffsetSweep(
    offset_dist=[
        dist.Dist(RNG.uniform, low=0, high=0),  # works better with experiment analysis than Dist(0)
        dist.Dist(RNG.uniform, low=-25, high=25),
        dist.Dist(RNG.uniform, low=-50, high=50),
        dist.Dist(RNG.uniform, low=-75, high=75),
        dist.Dist(RNG.uniform, low=-100, high=100),
    ]
)
