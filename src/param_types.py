from dataclasses import dataclass
from typing import Callable, Final, Union
import numpy as np

# "|", for example int|float requires python 3.10 or greater
class Dist:
    """define distributions for random drawing"""
    def __init__(self, dist: Union[int, float, Callable], *args, **kwargs):
        if isinstance(dist, Callable):
            rng = np.random.default_rng() # no seed needed since not used to draw
            # rng1.uniform != rng2.uniform, therefore must use name
            assert dist.__name__ in [rng.uniform.__name__, rng.normal.__name__], "unsupported distribution"
            
            self.__dist = dist
            self.__args = args
            self.__kwargs = kwargs
        elif isinstance(dist, int|float):
            self.__dist = self.__callable_const
            self.__args = (float(dist),)
            self.__kwargs = kwargs  # {"const": float(dist)}
                                    # either args or kwargs could be used here
    
    def __callable_const(self, const) -> float:
        return const

    def draw(self) -> float:
        return self.__dist(*self.__args, **self.__kwargs)

@dataclass
class PythonSignalRandArgs:
    """define the distribution from which deterministic parameters are drawn"""
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
    """define a python signal with deterministic parameters"""
    duration: float # specify either duration OR samples, let other be None
    samples: int
    freq: float # frequency
    amplitude: float # amplitude
    weight: float # weight is a factor that scales amplitude
    phase: float # phase shift
    offset_fctr: float  # offset_fctr is a factor proportional to amplitude and weight
                        # defining some offset
    sampling_rate: int

# TODO: type for algorithm and it's parameters

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
