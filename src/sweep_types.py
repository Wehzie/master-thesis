from abc import ABC
from dataclasses import dataclass
from typing import List, Union
import param_types as party
import algo

@dataclass
class AlgoWithArgs:
    Algo: algo.SearchAlgo # list of algorithms
    algo_args: party.AlgoArgs # list of arguments for each algorithm, in order with algos

@dataclass
class AlgoSweep:
    """repeat experiments over multiple algorithms
    
    produces a mean rmse, standard deviation and number of operations (z_ops) for a given configuration"""
    algo_with_args: List[AlgoWithArgs]
    m_averages: int            # number of averages for each experimental configuration



@dataclass
class ConstTimeSweep(ABC):
    """sweeps of PythonRandSignalArgs where time complexity between experiments is constant"""
    pass

@dataclass
class FreqSweep(ConstTimeSweep):
    freq_dist: List[party.Dist]

@dataclass
class AmplitudeSweep(ConstTimeSweep):
    amplitude: List[float]

@dataclass
class WeightSweep(ConstTimeSweep):
    weight_dist: List[party.WeightDist]

@dataclass
class PhaseSweep(ConstTimeSweep):
    phase_dist: List[party.Dist]

@dataclass
class OffsetSweep(ConstTimeSweep):
    offset_dist: List[party.Dist]



@dataclass
class ExpoTimeSweep:
    """sweeps of PythonRandSignalArgs where time complexity between experiments is worse then constant, mostly exponential"""
    pass

@dataclass
class NOscSweep(ExpoTimeSweep):
    n_osc: List[int]                    # number of n-oscillators

@dataclass
class ZOpsSweep(ExpoTimeSweep): # also expo time, in algo_args, not rand_args
    max_z_ops: List[int]                # maximum number of z-operations

@dataclass
class NumSamplesSweep(ExpoTimeSweep):
    samples: List[float]   # number of samples in the target signal