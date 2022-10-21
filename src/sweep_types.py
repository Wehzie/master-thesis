from dataclasses import dataclass
from typing import List
import param_types as party
import algo

@dataclass
class ConstTimeSweep:
    """sweeps of PythonRandSignalArgs where time complexity between experiments is constant"""
    f_dist: List[party.Dist]
    amplitude: List[float]
    weight_dist: List[party.Dist]
    phase_dist: List[party.Dist]

@dataclass
class ExpoTimeSweep:
    """sweeps of PythonRandSignalArgs where time complexity between experiments is worse then constant, mostly exponential"""
    n_osc: List[int]                    # number of oscillators

@dataclass
class SamplingRateSweep:
    sampling_rate_factor: List[float]   # factors to downsample the target signal

@dataclass
class AlgoSweep:
    """repeat experiments over multiple algorithms
    
    produces a mean rmse, standard deviation and number of operations (z_ops) for a given configuration"""
    algo: List[algo.SearchAlgo]    # list of algorithms
    algo_args: List[party.AlgoArgs]  # list of arguments for each algorithm, in order with algos
    m_averages: int            # number of averages for each experimental configuration
