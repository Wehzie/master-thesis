from abc import ABC
from dataclasses import dataclass
from typing import List
import dist
import algo
import algo_args_types as algarty

@dataclass
class AlgoWithArgs:
    Algo: algo.SearchAlgo # list of algorithms
    algo_args: algarty.AlgoArgs # list of arguments for each algorithm, in order with algos

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

# adding meta info to the sweep class is a good idea
# but requires changing the experimenteur class which iterates over fields of the sweep class
@dataclass
class FreqSweep(ConstTimeSweep):
    freq_dist: List[dist.Dist]

@dataclass
class AmplitudeSweep(ConstTimeSweep):
    amplitude: List[float]

@dataclass
class WeightSweep(ConstTimeSweep):
    weight_dist: List[dist.WeightDist]

@dataclass
class PhaseSweep(ConstTimeSweep):
    phase_dist: List[dist.Dist]

@dataclass
class OffsetSweep(ConstTimeSweep):
    offset_dist: List[dist.Dist]



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