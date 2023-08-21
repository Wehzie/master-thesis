"""This module defines valid arguments for signal generator modules."""

from dataclasses import dataclass
from typing import List, Union
from enum import Enum

import numpy as np

import dist

#### #### #### #### PYTHON SIGNALS #### #### #### ####


@dataclass
class PythonSignalRandArgs:
    """
    define the distribution from which deterministic parameters are drawn

    produces a signal matrix as a result
    a signal matrix is a circuit of n oscillators
    """

    description: str  # description of the parameter configuration

    duration: float  # signal duration in seconds
    samples: int  # number of samples in a signal
    sampling_rate: int  # number of samples per second

    n_osc: int  # number of oscillators

    amplitude: float  # amplitude of an oscillator (without gain/weight applied)
    freq_dist: dist.Dist  # random variable for the frequency of an oscillator
    weight_dist: dist.WeightDist  # the weight is applied to the amplitude; random variable to draw form
    phase_dist: dist.Dist  # random variable for the phase shift of an oscillator
    offset_dist: dist.Dist  # random variable for offset of an oscillator

    def get_time(self) -> np.ndarray:
        """get a time array either from the provided duration or number of samples"""
        if self.duration:
            time = np.arange(0, self.duration, 1 / self.sampling_rate)[0 : self.samples]
        else:  # samples given
            duration = self.samples / self.sampling_rate
            time = np.linspace(0, duration, self.samples, endpoint=False)
        return time

    def get_sample_spacing(self) -> float:
        """get the sample spacing of the signal"""
        return 1 / self.sampling_rate


@dataclass
class PythonSignalDetArgs:
    """
    define a python signal with deterministic parameters

    the signal is not weighted and has zero offset
    produces a single oscillator as result
    """

    duration: float  # specify either duration OR samples, let other be None
    samples: int  # length of the signal in number of samples
    freq: float  # frequency
    amplitude: float  # amplitude
    phase: float  # phase shift
    sampling_rate: int


#### #### #### #### SPICE SIGNALS #### #### #### ####


# class syntax
class SpipyGeneratorMode(Enum):
    """define the mode in which the signal generator operates"""

    SPICE = 1  # fully generate signals in SPICE
    EXTRAPOLATE = (
        2  # generate a short signal in SPICE and extrapolate it to the full duration in Python
    )
    CACHE = 3  # use cached SPICE signals and extrapolate them to the full duration in Python


@dataclass
class SpiceSumRandArgs:
    # TODO: rename SpipySumRandArgs
    """define distributions from which electric components are initialized"""

    description: str  # description of the parameter configuration

    n_osc: int  # number of oscillators
    v_in: float  # input voltage

    r_last: float  # resistance of resistor after summation of resistors
    r_control: float  # # resistor following the control terminal, doesn't affect oscillation
    # FIXME r_control ends up in dataframe, but we need r_dist

    r_dist: dist.Dist  # distribution for main resistor
    c_dist: dist.Dist  # distribution for main capacitor

    time_step: float  # simulated time step in seconds
    time_stop: float  # simulation stop time in seconds
    time_start: float  # simulation start time in seconds

    dependent_component: str  # a quantity and point to measure
    # for example v(A) or i(A)
    # where A is a node name
    # and v is voltage and i is current

    phase_dist: dist.Dist  # shift SPICE signal by phase, in radians, inject phase into netlist via time_start
    weight_dist: dist.WeightDist  # scale amplitude ot SPICE signal in Python
    offset_dist: dist.Dist  # alter offset of SPICE signal in Python

    generator_mode: SpipyGeneratorMode  # method of generating the signal
    down_sample_factor: Union[int, None] = None  # down sample the generated signal by this factor

    def estimate_number_of_samples(self) -> int:
        """estimate the number samples that SPICE will produce given the rand_args parameters"""
        num_samples_float = (self.time_stop - self.time_start) / self.time_step
        num_samples = np.around(num_samples_float).astype(int)

        if self.down_sample_factor is None:
            return num_samples

        num_samples_after_downsampling = int(num_samples * self.down_sample_factor)
        return num_samples_after_downsampling

    def get_sampling_rate(self) -> int:
        """get the sampling rate of the signal"""
        if self.down_sample_factor is None:
            return np.around(1 / self.time_step).astype(int)

        spice_sampling_rate = 1 / self.time_step
        sampling_rate = np.around(spice_sampling_rate * self.down_sample_factor).astype(int)
        return sampling_rate

    def get_sample_spacing(self) -> float:
        """get the sample spacing of the signal"""
        if self.down_sample_factor is None:
            return self.time_step

        sample_spacing = self.time_step / self.down_sample_factor
        return sample_spacing

    def get_duration(self) -> float:
        """get the duration of the signal"""
        return self.time_stop - self.time_start


@dataclass
class SpiceSingleDetArgs:
    """define deterministic electric components for a single oscillator circuit"""

    n_osc: int  # number of oscillators
    v_in: float  # input voltage, influences frequency, and offset
    r: float  # resistor controls frequency, doesn't affect amplitude
    r_last: float  # for a single oscillator the effect of r_last is equal to adding the resistance to r
    r_control: float  # resistor following the control terminal, doesn't affect oscillation
    c: float  # capacitor, impact similar to main resistor
    time_step: float  # simulated time step in seconds
    time_stop: float  # simulation stop time in seconds
    time_start: float  # simulation start time in seconds
    dependent_component: str = "v(osc1)"  # the node to read out and report back to Python
    phase: float = 0.0  # phase shift in radians, added on Python side
    generator_mode: SpipyGeneratorMode = SpipyGeneratorMode.CACHE
    down_sample_factor: Union[
        int, None
    ] = None  # down sample the generated signal by this factor, None means no downsampling
    #   a value of 0.1 means that 1 sample is taken every 10 samples


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
