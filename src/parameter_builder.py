"""This module provides functions to bundle parameters for experiments."""

from pathlib import Path
from typing import List, Final, Tuple

import gen_signal_args_types as party

import dist
import data_preprocessor
import data_io
import const


def estimate_summed_offset(
    v_in: float, n_osc: int, weight_dist_low: float, weight_dist_high: float
) -> float:
    """Estimate the offset resulting from summing the voltages of n oscillators."""
    weight_dist_mean = (weight_dist_low + weight_dist_high) / 2
    one_sided_offset = (
        v_in * n_osc * weight_dist_mean * 1.14
    )  # roughly the offset needed to get mean=0
    return one_sided_offset


def comp_loc(low: float, high: float) -> float:
    """compute loc (mean) for normal distribution from uniform distribution's low and high bounds"""
    return low + (high - low) / 2


def comp_scale(low: float, high: float) -> float:
    """compute scale (std dev) for normal distribution from a uniform distribution's low and high bounds"""
    return (high - low) / 2


def append_normal(
    uniform_li: List[dist.Dist], only_uniform: bool = True, only_normal: bool = False
) -> List[dist.Dist]:
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
            raise ValueError(
                "experiments with constant distributions aren't supported, do low=const, high=const instead"
            )
        loc = comp_loc(d.kwargs["low"], d.kwargs["high"])
        scale = comp_scale(d.kwargs["low"], d.kwargs["high"])
        norm_li.append(DistType(const.RNG.normal, loc=loc, scale=scale, n=d.n))
    if only_normal:
        return norm_li
    return uniform_li + norm_li


def init_target2rand_args(
    rand_args: party.PythonSignalRandArgs, scale_factor: float = 0.5
) -> Tuple[party.PythonSignalRandArgs, Tuple]:
    """load, downsample target and inject number of samples into rand_args"""
    # loading and manipulating the target signal
    raw_sampling_rate, raw_target, raw_dtype = data_io.load_data()
    target_full_len: Final = data_preprocessor.downsample_by_factor_typesafe(
        raw_target, scale_factor
    )
    # shorten the target
    target: Final = data_preprocessor.take_middle_third(target_full_len)
    # save to wav
    sampling_rate = int(scale_factor * raw_sampling_rate)
    data_io.save_signal_to_wav(
        target, sampling_rate, raw_dtype, Path("data/target_downsampled.wav")
    )
    # init search params
    rand_args = rand_args
    rand_args.samples = len(target)  # generated signals match length of target
    # NOTE: the sampling rate could also be set lower instead
    return rand_args, (sampling_rate, target, raw_dtype)
