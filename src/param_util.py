"""utility functions for parameter generation"""
import numpy as np

from pathlib import Path
from typing import List, Final, Tuple

from algo import SearchAlgo
import algo_las_vegas as alave
import algo_monte_carlo as almoca
import algo_evolution as alevo
import algo_gradient as algra
import algo_mcmc as almcmc
import algo_args_types as algarty
import param_types as party
import sweep_types as sweety
import gen_signal as gen_signal
import gen_signal_python as gensi_python
import param_mask
import meta_target

import dist
import data_preprocessor
import data_io
import const
import param_types as party


def comp_loc(low: float, high: float) -> float:
    """compute loc (mean) for normal distribution from uniform distribution's low and high bounds"""
    return low + (high - low) / 2


def comp_scale(low: float, high: float) -> float:
    """compute scale (std dev) for normal distribution from a uniform distribution's low and high bounds"""
    return (high - low) / 2


def append_normal(uniform_li: List[dist.Dist], only_uniform: bool = True, only_normal: bool = False) -> List[dist.Dist]:
    """given a list of uniform distributions, compute the corresponding normal distributions and append them to the list
    
    args:
        uniform_li: list of uniform distributions
        only_uniform: if True, return uniform_li without appending normal distributions
        only_normal: if True, return only normal distributions
    """
    if only_uniform:
        return uniform_li
    DistType = type(uniform_li[0])
    norm_li = list()
    for d in uniform_li:
        if d.is_const():
            raise ValueError("experiments with constant distributions aren't supported, do low=const, high=const instead")
        loc = comp_loc(d.kwargs["low"], d.kwargs["high"])
        scale = comp_scale(d.kwargs["low"], d.kwargs["high"])
        norm_li.append(DistType(const.RNG.normal, loc=loc, scale=scale, n=d.n))
    if only_normal:
        return norm_li
    return uniform_li + norm_li


def init_target2rand_args(rand_args: party.PythonSignalRandArgs, scale_factor: float = 0.5) -> Tuple[party.PythonSignalRandArgs, Tuple]:
    """load, downsample target and inject number of samples into rand_args"""
    # loading and manipulating the target signal
    raw_sampling_rate, raw_target, raw_dtype = data_io.load_data()
    target_full_len: Final = data_preprocessor.downsample_by_factor_typesafe(raw_target, scale_factor)
    # shorten the target
    target: Final = data_preprocessor.take_middle_third(target_full_len)
    # save to wav
    sampling_rate = int(scale_factor*raw_sampling_rate)
    data_io.save_signal_to_wav(target, sampling_rate, raw_dtype, Path("data/target_downsampled.wav"))
    # init search params
    rand_args = rand_args
    rand_args.samples = len(target) # generated signals match length of target
                                    # NOTE: the sampling rate could also be set lower instead
    return rand_args, (sampling_rate, target, raw_dtype)


def init_algo_sweep(meta_target: meta_target.MetaTarget, rand_args: party.PythonSignalRandArgs,
sig_generator: gen_signal.SignalGenerator = gensi_python.PythonSigGen(),
max_z_ops: int = 3e3, m_averages: int = 2, test_mode: bool = False) -> sweety.AlgoSweep:
    """initialize a set of algorithms with varying arguments for a fixed target and a fixed set of rand_args from which to draw oscillators
    
    args:
        target: the target signal
        rand_args: the random variables from which oscillators are drawn
        z: the maximum number of z-operations to perform
        m_averages: the number of times to average the results of each algorithm"""
    one_shot_algos = [
        sweety.AlgoWithArgs(
            almoca.MCOneShot,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCOneShotWeight,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
    ]
    exploit_algos = [
        sweety.AlgoWithArgs(
            almoca.MCExploit,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCExploitJ10,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=10, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCExploitWeight,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCExploitNeighborWeight,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCExploitDecoupled,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCExploitFast,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
    ]
    grow_shrink_algos = [
        sweety.AlgoWithArgs(
            almoca.MCGrowShrink,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, l_damp_prob=0.5, h_damp_fac=0.5, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCDampen,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, h_damp_fac=0.5, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCPurge,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, h_damp_fac=0, sig_generator=sig_generator),
        ),
    ]
    oscillator_anneal_algos = [
        sweety.AlgoWithArgs(
            almoca.MCOscillatorAnneal,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCOscillatorAnnealWeight,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCOscillatorAnnealLog,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almoca.MCOscillatorAnnealLogWeight,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
    ]
    las_vegas_algos = [
        sweety.AlgoWithArgs(
            alave.LasVegas,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            alave.LasVegasWeight,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
    ]
    population_algos = [
        sweety.AlgoWithArgs(
            alevo.DifferentialEvolution,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
    ]
    mcmc_algos = [
        sweety.AlgoWithArgs(
            almcmc.MCExploitErgodic,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almcmc.MCExploitAnneal,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almcmc.MCExploitAnnealWeight,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almcmc.BasinHopping,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almcmc.ScipyAnneal,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
        sweety.AlgoWithArgs(
            almcmc.ScipyDualAnneal,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, sig_generator=sig_generator),
        ),
    ]
    gradient_algos = [
        sweety.AlgoWithArgs(
            algra.LinearRegression,
            algarty.AlgoArgs(rand_args, meta_target, sig_generator=sig_generator),
        ),
    ]

    all_algos_with_args: Final = one_shot_algos + exploit_algos + grow_shrink_algos + oscillator_anneal_algos + las_vegas_algos + population_algos + mcmc_algos + gradient_algos
    if test_mode:
        return sweety.AlgoSweep(all_algos_with_args, m_averages)

    some_algos_with_args = one_shot_algos# + exploit_algos + grow_shrink_algos + oscillator_anneal_algos + las_vegas_algos + population_algos + mcmc_algos + gradient_algos
    single_algo_with_args = [
        sweety.AlgoWithArgs(
            almoca.MCExploit,
            algarty.AlgoArgs(rand_args, meta_target, max_z_ops, j_replace=1, sig_generator=sig_generator),
        ),
    ]
    
    return sweety.AlgoSweep(all_algos_with_args, m_averages, param_mask.algo_masks)

algo_list: List[SearchAlgo] = [
    almoca.MCExploit,
    almoca.MCExploitJ10,
    almoca.MCExploitDecoupled,
    almoca.MCExploitWeight,
    almoca.MCExploitFast,
    almoca.MCExploitNeighborWeight,
    almoca.MCOneShot,
    almoca.MCOneShotWeight,
    almoca.MCGrowShrink,
    almoca.MCDampen,
    almoca.MCPurge,
    almoca.MCOscillatorAnneal,
    almoca.MCOscillatorAnnealWeight,
    almoca.MCOscillatorAnnealLog,
    almoca.MCOscillatorAnnealLogWeight,
    alave.LasVegas,
    alave.LasVegasWeight,
    alevo.DifferentialEvolution,
    almcmc.MCExploitErgodic,
    almcmc.MCExploitAnneal,
    almcmc.MCExploitAnnealWeight,
    almcmc.BasinHopping,
    almcmc.ScipyAnneal,
    almcmc.ScipyDualAnneal,
    algra.LinearRegression,
]