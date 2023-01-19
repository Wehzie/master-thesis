from dataclasses import dataclass
from typing import List

import algo_args_types as algarty
import sample

@dataclass
class ResultSweep: # TODO: rename ResultOfSweep
    """running an experiment with AlgoSweep args produces a list of ResultSweep"""
    algo_name: str
    algo_args: algarty.AlgoArgs       # to track what's going on at any point in time
    mean_rmse: float
    std_rmse: float
    mean_z_ops: float
    std_z_ops: float
    m_averages: int                 # number of averages taken

@dataclass
class ResultAlgoCallback:
    """result of a running algorithm created from a callback while running"""
    best_sample: sample.Sample
    z_ops: int
