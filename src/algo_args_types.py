from dataclasses import dataclass
from typing import List, Union
from pathlib import Path

import numpy as np

import gen_signal
import gen_signal_python as gensi_python
import const
import param_types as party
import meta_target

@dataclass
class AlgoArgs:
    """
    produces a best sample and the number of operations used as a result
    
    a single sample is an optimized signal matrix and array of weights
    z_ops measures z initialized, drawn, discarded - oscillators or weights
    """

    rand_args: party.PythonSignalRandArgs   # arguments to init a signal matrix
    meta_target: meta_target.MetaTarget     # the target signal to optimize for (and its meta data)
    max_z_ops: Union[None, int]     = None  # maximum number of operations until learning is aborted
    k_samples: int                  = None  # number of times to re-run base algorithm
    j_replace: Union[None, int]    = None   # number of oscillators to replace in each iteration for MCExploit
    l_damp_prob: Union[None, float] = None  # dampening probability for MCGrowShrink
    h_damp_fac: Union[None, float]   = None # dampening factor for MCGrowShrink, MCDampen, MCPurge
    mp: bool = const.MULTIPROCESSING        # whether to use multiprocessing
    z_ops_callbacks: Union[None, List[int]] = None # at each value of z_ops store the best sample up to that point
    store_det_args: bool            = False # whether to store det_args for each k
    history: bool                   = False # whether to store each sample
    args_path: Union[None, Path]    = None  # whether to flush samples in RAM to file at given path
    # TODO: since sig_generator is important it should be an earlier argument
    sig_generator: gen_signal.SignalGenerator = gensi_python.PythonSigGen() # method of generating signals
