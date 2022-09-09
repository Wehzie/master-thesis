
from dataclasses import dataclass
from typing import Callable, Final
import numpy as np

rng = np.random.default_rng(seed=5)

class Dist:
    """define distributions for random drawing"""
    def __init__(self, dist: int|float|Callable, *args, **kwargs):
        if isinstance(dist, Callable):
            rng = np.random.default_rng()
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
    n_osc: int = 100 # number of oscillators

    # NOTE: specify either duration or samples while the other is none
    # TODO: handle this more appropriately with a separate Type/Class
    duration: float = None # signal duration in seconds
    samples: int = 300 # number of samples in a signal

    f_dist: Dist = Dist(rng.uniform, low=1e5, high=1e6)
    amplitude: float = 0.5 # resembling 0.5 V amplitude of V02
    weight_dist: Dist = Dist(rng.uniform, low=0.2, high=1)    # amplitude=weight_dist*default_amplitude
                                                            # resistor doesn't amplify so not > 1
    phase_dist: Dist = Dist(rng.uniform, low=-1/3, high=1/3) # phase=phase_dist.draw()*pi
    offset_dist: Dist = Dist(rng.uniform, low=-1/3, high=1/3) # offset=offset_dist.draw()*amplitude*weight
    sampling_rate: int = 11025 # the sampling rate of the Magpie signal

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
