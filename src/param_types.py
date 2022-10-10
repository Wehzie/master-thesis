from asyncore import file_dispatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, List, Union
import numpy as np
import algo

# "|", for example int|float requires python 3.10 or greater
class Dist:
    """define distributions for random drawing"""
    def __init__(self, dist: Union[int, float, Callable], *args, **kwargs):
        if isinstance(dist, Callable):
            rng = np.random.default_rng() # no seed needed since not used to draw
            # rng1.uniform != rng2.uniform, therefore must use name
            assert dist.__name__ in [rng.uniform.__name__, rng.normal.__name__], "unsupported distribution"
            
            self.dist = dist
            self.args = args
            self.kwargs = kwargs
        elif isinstance(dist, int|float):
            self.dist = self.callable_const
            self.args = (float(dist),)
            self.kwargs = kwargs  # {"const": float(dist)}
                                    # either args or kwargs could be used here

    def callable_const(self, const) -> float:
        return const

    def draw(self) -> float:
        return self.dist(*self.args, **self.kwargs)
    
    def __repr__(self) -> str:
        if isinstance(self.dist, Callable):
            dist = self.dist.__name__
        else:
            dist = str(self.dist)
        dist += ", "
        args = str(self.args) + ", "
        kwargs = str(self.kwargs)
        return dist + args + kwargs + "\n"

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

    f_dist: Dist # frequency distribution
    amplitude: float # shared by all oscillators in a sum
    weight_dist: Dist # amplitude=weight_dist*default_amplitude
                     
    phase_dist: Dist # phase=phase_dist.draw()*pi
    offset_dist: Dist # offset=offset_dist.draw()*amplitude*weight
    sampling_rate: int # number of samples per second

@dataclass
class PythonSignalDetArgs:
    """define a python signal with deterministic parameters
    
    produces a single oscillator as result
    """
    duration: float # specify either duration OR samples, let other be None
    samples: int
    freq: float # frequency
    amplitude: float # amplitude
    weight: float # weight is a factor that scales amplitude
    phase: float # phase shift
    offset_fctr: float  # offset_fctr is a factor proportional to amplitude and weight
                        # defining some offset
    sampling_rate: int

@dataclass
class AlgoArgs:
    """
    produces a best sample and the number of operations used as a result
    
    a single sample is an optimized signal matrix and array of weights
    z_ops measures z initialized, drawn, discarded - oscillators or weights
    """

    rand_args: int                  # arguments to init a signal matrix
    target: np.ndarray              # the target to optimize for
    weight_init: Union[None, str]   # initialization of weight array
    max_z_ops: int                  # maximum number of operations until learning is aborted
    k_samples: int                  # number of times to re-run base algorithm
    j_exploit: Union[None, int]     # within-model exploit iterations for monte-carlo algorithms
    store_det_args: bool            # whether to store det_args for each k
    history: bool                   # whether to store each sample
    args_path: Path                 # whether to flush samples in RAM to file at given path

@dataclass
class SweepConstTimeArgs:
    """sweeps of PythonRandSignalArgs where time complexity between experiments is constant"""
    f_dist: List[Dist]
    amplitude: List[float]
    weight_dist: List[Dist]
    phase_dist: List[Dist]

@dataclass
class SweepExpoTimeArgs:
    """sweeps of PythonRandSignalArgs where time complexity between experiments is worse then constant, mostly exponential"""
    n_osc: List[int]                    # number of oscillators
    sampling_rate_factor: List[float]   # factors to downsample the target signal 

@dataclass
class SweepAlgos:
    """repeat experiments over multiple algorithms
    
    produces a mean rmse, standard deviation and number of operations (z_ops) for a given configuration"""
    algo: List    # list of algorithms
    algo_args: List[AlgoArgs]  # list of arguments for each algorithm, in order with algos
    m_averages: int            # number of averages for each experimental configuration

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
