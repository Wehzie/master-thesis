"""
This module defines the ExperimentMask class.

An ExperimentMask is a collection of algorithms which are compared against each other.
A mask is applied after an experiment has run and allows generating plots with different subsets of algorithms.
The main intent is to reduce clutter in plots by grouping algorithms into categories.
"""

from dataclasses import dataclass
from typing import List, Union

import matplotlib.colors as mcolors

TABLEAU_COLORS: List[str] = [color for color in list(mcolors.TABLEAU_COLORS.values())]  # HEX colors

import algo


@dataclass
class ExperimentMask:
    """each mask compares a subset of algorithms against another subset"""

    filename: str
    title: str  # title for plots
    description: Union[str, None]
    algo_groups: List[List[algo.SearchAlgo]]

    def get_color_map(self) -> dict:
        """get matplotlib compatible color map for each algorithm in this mask"""
        color_map = {}
        for algo_group, color in zip(self.algo_groups, TABLEAU_COLORS):
            for algorithm in algo_group:
                color_map[algorithm] = color
        return color_map

    def get_algo_names(self) -> List[str]:
        """form flat list with names of all algorithms in this mask"""
        names = []
        for algo_group in self.algo_groups:
            for algorithm in algo_group:
                names.append(algorithm)
        return names
