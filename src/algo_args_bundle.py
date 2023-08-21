"""This module implements a class that bundles together an algorithm and its arguments."""

import algo
import algo_args_type as algarty
from dataclasses import dataclass


@dataclass
class AlgoWithArgs:
    """pair an algorithm with its arguments"""

    Algo: algo.SearchAlgo  # list of algorithms
    algo_args: algarty.AlgoArgs  # list of arguments for each algorithm, in order with algos
