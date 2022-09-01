
from dataclasses import dataclass
from typing import List


@dataclass
class PythonSignalRandArgs:
    """define the distribution from which deterministic parameters are drawn"""
    n_osc: int = 1000 # number of oscillators
    f_lo: float = 1e5 # frequency bounds
    f_hi: float = 1e6
    # NOTE: specify either duration or samples while the other is none
    # TODO: handle this more appropriately with a separate Type/Class
    duration: float = None # signal duration in seconds
    samples: int = 300 # number of samples in a signal
    weight: str = "random" # sawtooth shape
                            # TODO: set const
    random_phase: bool = True

@dataclass
class PythonSignalDetArgs:
    """define a python signal with deterministic parameters"""
    freq: float
    duration: float
    samples: int
    weight: float
    random_phase: bool

@dataclass
class SpiceSumRandArgs:
    n_osc: int = 1
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
    time_stop: float = 1e-5
    time_start: float = 0

    dependent_component: str = "v(osc1)"

@dataclass
class SpiceSumDetArgs:
    n_osc: int
    v_in: float
    r_list: List[float]
    r_last: float
    r_control: float
    c_list: List[float]

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
    dependent_component: str = "v(osc1)"
