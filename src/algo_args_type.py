"""this module defines the structure of arguments required by algorithms modules."""
from dataclasses import dataclass
from typing import List, Union
from pathlib import Path

import gen_signal
import const
import gen_signal_args_types as party
import meta_target


@dataclass
class AlgoArgs:
    """
    produces a best sample and the number of operations used as a result

    a single sample is an optimized signal matrix and array of weights
    z_ops measures z initialized, drawn, discarded - oscillators or weights
    """

    sig_generator: gen_signal.SignalGenerator  # method of generating signals
    rand_args: party.UnionRandArgs  # arguments to init a signal matrix
    meta_target: meta_target.MetaTarget  # the target signal to optimize for (and its meta data)
    max_z_ops: Union[None, int] = None  # maximum number of operations until learning is aborted
    k_samples: int = None  # number of times to re-run base algorithm
    j_replace: Union[
        None, int
    ] = None  # number of oscillators to replace in each iteration for MCExploit
    l_damp_prob: Union[None, float] = None  # dampening probability for MCGrowShrink
    h_damp_fac: Union[None, float] = None  # dampening factor for MCGrowShrink, MCDampen, MCPurge
    mp: bool = const.MULTIPROCESSING  # whether to use multiprocessing
    z_ops_callbacks: Union[
        None, List[int]
    ] = None  # at each value of z_ops store the best sample up to that point
    store_det_args: bool = False  # whether to store det_args for each k
    history: bool = False  # whether to store each sample
    args_path: Union[None, Path] = None  # whether to flush samples in RAM to file at given path
