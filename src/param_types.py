from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, List, Union
import numpy as np
from dist import Dist, WeightDist
import const

@dataclass
class PythonSignalRandArgs:
    """define the distribution from which deterministic parameters are drawn
    
    produces a signal matrix as a result
    a signal matrix is a circuit of n oscillators
    """
    n_osc: int # number of oscillators

    # NOTE: specify either duration or samples while the other is none
    # TODO: handle this more appropriately with a separate Type/Class
    duration: float # signal duration in seconds
    samples: int # number of samples in a signal

    freq_dist: Dist # frequency distribution
    amplitude: float # shared by all oscillators in a sum
    weight_dist: WeightDist # amplitude=weight_dist*default_amplitude
                     
    phase_dist: Dist # phase=phase_dist.draw()*pi
    offset_dist: Dist # offset=offset_dist.draw()*amplitude*weight
    sampling_rate: int # number of samples per second

@dataclass
class PythonSignalDetArgs:
    """define a python signal with deterministic parameters
    the signal is not weighted and has zero offset
    
    produces a single oscillator as result
    """
    duration: float # specify either duration OR samples, let other be None
    samples: int # length of the signal in number of samples
    freq: float # frequency
    amplitude: float # amplitude
    phase: float # phase shift
    sampling_rate: int

@dataclass
class AlgoArgs:
    """
    produces a best sample and the number of operations used as a result
    
    a single sample is an optimized signal matrix and array of weights
    z_ops measures z initialized, drawn, discarded - oscillators or weights
    """

    rand_args: PythonSignalRandArgs         # arguments to init a signal matrix
    target: np.ndarray                      # the target to optimize for
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


@dataclass
class SpiceSumRandArgs:
    n_osc: int = 2
    v_in: float = 14
    
    r_last: float = 0
    r_control: float = 1e6
    r_min: float = 30e3
    r_max: float = 70e3
    r_dist: str = "uniform"

    c_min: float = 300e-12
    c_max: float = 300e-12
    c_dist: str = "uniform"

    time_step: float = 5e-9
    time_stop: float = 3.3e-2
    time_start: float = 0

    dependent_component: str = "v(osc1)"

@dataclass
class SpiceSumDetArgs:
    n_osc: int
    v_in: float
    r_list: list[float]
    r_last: float
    r_control: float
    c_list: list[float]

@dataclass
class SpiceSingleDetArgs:
    n_osc: int
    v_in: float
    r: float
    r_last: float
    r_control: float
    c: float
    time_step: float
    time_stop: float
    time_start: float
    sim_success: bool = None
    dependent_component: str = "v(osc1)" # TODO: is this being used?
