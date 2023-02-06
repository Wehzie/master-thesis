"""Parameters for the SpiPy (hybrid) signal generator."""
# TODO: reduce code duplication with params_hybrid_test.py

import param_types as party
import dist
import const
import sweep_types as sweety
RNG = const.RNG

# parameters for testing purposes
spice_single_det_args = party.SpiceSingleDetArgs(
    n_osc=1,
    v_in=14,
    r=140e3,
    r_last=1,
    r_control=1,
    c=300e-12,
    time_step=2e-9,
    time_stop=1e-3,
    time_start=0,
    dependent_component="v(osc1)",
    phase=0.5,
    extrapolate=True,
    down_sample_factor=1/10,
)

spice_rand_args_n_osc = 10
v_in = 14
weight_dist_low = 0
weight_dist_high = 10
weight_dist_mean = (weight_dist_high + weight_dist_low) / 2
one_sided_offset = v_in * spice_rand_args_n_osc * weight_dist_mean * 1.14 # roughly the offset needed to get mean=0

spice_rand_args_uniform = party.SpiceSumRandArgs(
    description="VO2-RC circuit parameters inspired by Maffezzoni et al. 2015",
    n_osc=spice_rand_args_n_osc,
    v_in=v_in,

    r_last=1,
    r_control=1,
    r_dist=dist.Dist(RNG.uniform, low=20e3, high=140e3),
    c_dist=dist.Dist(300e-12),

    time_step=2e-9,
    time_stop=1e-3,
    time_start=0,

    dependent_component="v(osc1)",

    # Python controlled parameters
    phase_dist = dist.Dist(RNG.uniform, low=0, high=2),
    weight_dist = dist.WeightDist(RNG.uniform, low=weight_dist_low, high=weight_dist_high, n=spice_rand_args_n_osc),
    offset_dist = dist.Dist(RNG.uniform, low=0, high=0),

    # runtime and memory optimizations
    extrapolate=True,
    down_sample_factor=1/100,
)

n_osc_sweep = sweety.NOscSweep(
    n_osc=[50, 100, 200, 500, 1000],
)


z_ops_sweep = sweety.ZOpsSweep(
    max_z_ops=[0, 5e2, 1e3, 5e3, 1e4, 5e4],
)

sampling_rate_sweep = sweety.NumSamplesSweep([50, 100, 300, 500])
