from abc import ABC
from dataclasses import dataclass
from typing import List, Union
import dist
import algo
import algo_args_types as algarty
import param_mask

@dataclass
class AlgoWithArgs:
    Algo: algo.SearchAlgo # list of algorithms
    algo_args: algarty.AlgoArgs # list of arguments for each algorithm, in order with algos

@dataclass
class AlgoSweep:
    """
    repeat experiments over multiple algorithms.
    
    produces a mean rmse, standard deviation and number of operations (z_ops) for a given configuration.
    """
    algo_with_args: List[AlgoWithArgs]
    m_averages: int            # number of averages for each experimental configuration

    # list of groups of algorithms, that are interesting to compare
    # for example
    # gradient based vs random search based algorithms
    # each 
    #   the purpose is
    #   1) plotting the results of each group in a different color
    #   2) averaging the results of each group
    #   3) plotting the results of each group separately
    algo_masks: Union[None,
                 List[param_mask.ExperimentMask]] = None


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
class ExpoTimeSweep(ABC):
    """sweeps of PythonRandSignalArgs where time complexity between experiments is worse then constant, mostly exponential"""
    NotImplemented
    # TODO: using fixed field names I can replace
    #   for val_schedule in fields(sweep_args)
    #       for awa in algo_sweep.algo_with_args:
    #           for val in getattr(sweep_args, val_schedule):
    #               ...
    # in the experimenteur class module with
    #   for awa in algo_sweep.algo_with_args:
    #       for val in sweep_args.val_schedule:
    #           ...
    # filename: str                       # filename for saving the results
    # iv_name: str                        # type of the independent variable
    # val_schedule: List                  # values of the independent variable
    # dv_name: str                        # type of the dependent variable

@dataclass
class NOscSweep(ExpoTimeSweep):
    n_osc: List[int]                    # number of n-oscillators

@dataclass
class ZOpsSweep(ExpoTimeSweep): # also expo time, in algo_args, not rand_args
    max_z_ops: List[int]                # maximum number of z-operations

@dataclass
class NumSamplesSweep(ExpoTimeSweep):
    samples: List[float]   # number of samples in the target signal