
from dataclasses import dataclass


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