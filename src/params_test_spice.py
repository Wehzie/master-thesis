import param_types as party
import dist
import const
RNG = const.RNG


spice_single_det_args = party.SpiceSingleDetArgs(
    n_osc=1,
    v_in=14,
    r=80e3,
    r_last=1,
    r_control=1,
    c=300e-12,
    time_step=5e-9,
    time_stop=1e-5,
    time_start=0,
    dependent_component="v(osc1)",
    phase=0.5,
)

spice_rand_args_n_osc = 5
v_in = 14
weight_dist_low = 0
weight_dist_high = 10
weight_dist_mean = (weight_dist_high + weight_dist_low) / 2
one_sided_offset = v_in * spice_rand_args_n_osc * weight_dist_mean * 1.14 # roughly the offset needed to get mean=0

spice_rand_args_uniform = party.SpiceSumRandArgs(
    n_osc=spice_rand_args_n_osc,
    v_in=v_in,

    r_last=1,
    r_control=1,
    r_dist=dist.Dist(RNG.uniform, low=20e3, high=70e3),
    c_dist=dist.Dist(300e-12),

    time_step=5e-9,
    time_stop=1e-4,
    time_start=0,

    dependent_component="v(osc1)",

    # Python controlled parameters
    phase_dist = dist.Dist(RNG.uniform, low=0, high=2),
    weight_dist = dist.WeightDist(RNG.uniform, low=weight_dist_low, high=weight_dist_high, n=spice_rand_args_n_osc),
    offset_dist = dist.Dist(-one_sided_offset), 
    # offset_dist = dist.Dist(RNG.uniform, low=-one_sided_offset, high=one_sided_offset),
)
