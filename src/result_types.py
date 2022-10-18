from dataclasses import dataclass

import param_types as party
import sample

@dataclass
class ResultAlgoSweep:
    """running an experiment with AlgoSweep args produces a list of ResultAlgoSweep"""
    algo_name: str
    algo_args: party.AlgoArgs       # to track what's going on at any point in time
    mean_rmse: float
    std_rmse: float
    mean_z_ops: float
    std_z_ops: float

@dataclass
class ResultAlgoCallback:
    """result of a running algorithm created from a callback while running"""
    best_sample: sample.Sample
    z_ops: int
