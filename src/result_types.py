"""This module implements a dataclass that describes the structure of the results following an experiment."""

from dataclasses import dataclass

import algo_args_type as algarty


@dataclass
class ResultSweep:  # TODO: rename ResultOfSweep
    """running an experiment with AlgoSweep args produces a list of ResultSweep"""

    algo_name: str
    algo_args: algarty.AlgoArgs  # to track what's going on at any point in time
    mean_rmse: float
    std_rmse: float
    mean_z_ops: float
    std_z_ops: float
    m_averages: int  # number of averages taken
