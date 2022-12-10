from dataclasses import dataclass
from typing import List, Union

import dist

#### #### #### #### PYTHON SIGNALS #### #### #### ####


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

    freq_dist: dist.Dist # frequency distribution
    amplitude: float # shared by all oscillators in a sum
    weight_dist: dist.WeightDist # amplitude=weight_dist*default_amplitude
                     
    phase_dist: dist.Dist # phase=phase_dist.draw()*pi
    offset_dist: dist.Dist # offset=offset_dist.draw()*amplitude*weight
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


#### #### #### #### SPICE SIGNALS #### #### #### ####


@dataclass
class SpiceSumRandArgs:
    # TODO: rename SpipySumRandArgs
    """define distributions from which electric components are initialized"""
    n_osc: int          # number of oscillators
    v_in: float         # input voltage
    
    r_last: float       # resistance of resistor after summation of resistors
    r_control: float    # # resistor following the control terminal, doesn't affect oscillation

    r_dist: dist.Dist        # distribution for main resistor
    c_dist: dist.Dist        # distribution for main capacitor

    time_step: float    # simulated time step in seconds
    time_stop: float    # simulation stop time in seconds
    time_start: float   # simulation start time in seconds

    dependent_component: str # a quantity and point to measure
                             # for example v(A) or i(A)
                             # where A is a node name
                             # and v is voltage and i is current

    phase_dist: dist.Dist        # shift SPICE signal by phase, in radians, inject phase into netlist via time_start
    weight_dist: dist.WeightDist # scale amplitude ot SPICE signal in Python
    offset_dist: dist.Dist       # alter offset of SPICE signal in Python

    down_sample_factor: Union[int, None] = None # down sample the generated signal by this factor

@dataclass
class SpiceSingleDetArgs:
    """define deterministic electric components for a single oscillator circuit"""
    n_osc: int          # number of oscillators
    v_in: float         # input voltage, influences frequency, and offset
    r: float            # resistor controls frequency, doesn't affect amplitude
    r_last: float       # for a single oscillator the effect of r_last is equal to adding the resistance to r
    r_control: float    # resistor following the control terminal, doesn't affect oscillation
    c: float            # capacitor, impact similar to main resistor
    time_step: float    # simulated time step in seconds
    time_stop: float    # simulation stop time in seconds
    time_start: float   # simulation start time in seconds
    dependent_component: str = "v(osc1)"        # the node to read out and report back to Python
    phase: float = 0.0                          # phase shift in radians, added on Python side
    down_sample_factor: Union[int, None] = None # down sample the generated signal by this factor, None means no downsampling
                                                # a value of 0.1 means that 1 sample is taken every 10 samples


@dataclass
class SpiceSumDetArgs:
    """define deterministic electric components for a circuit of parallel oscillators"""
    n_osc: int
    v_in: float
    r_list: List[float]
    r_last: float
    r_control: float
    c_list: List[float]


#### #### #### #### SIGNAL PARAMETER UNIONS #### #### #### ####


UnionRandArgs = Union[PythonSignalRandArgs, SpiceSumRandArgs]
UnionDetArgs = Union[PythonSignalDetArgs, SpiceSingleDetArgs, SpiceSumDetArgs]