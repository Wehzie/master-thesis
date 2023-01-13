"""
set of masks over the experiment results
each mask contains clusters of algorithms which are compared
"""

from dataclasses import dataclass
from typing import List, Union

import matplotlib.colors as mcolors
TABLEAU_COLORS: List[str] = [color for color in list(mcolors.TABLEAU_COLORS.values())] # HEX colors

import algo

@dataclass
class ExperimentMask:
    """each mask compares a subset of algorithms against another subset"""
    filename: str
    title: str      # title for plots
    description: Union[str, None]
    algo_groups: List[List[algo.SearchAlgo]]

    def get_color_map(self) -> dict:
        """get matplotlib compatible color map for each algorithm in this mask"""
        color_map = {}
        for algo_group, color in zip(self.algo_groups, TABLEAU_COLORS):
            for algo in algo_group:
                color_map[algo] = color
        return color_map

    def get_algo_names(self) -> List[str]:
        """form flat list with names of all algorithms in this mask"""
        names = []
        for algo_group in self.algo_groups:
            for algo in algo_group:
                names.append(algo)
        return names

m1 = ExperimentMask(
    "full_vs_weight_mcexploit",
    "full optimization vs. weight optimization",
    "full optimization vs. weight optimization by example of MCExploit algorithms",
    [
        [
            "MCExploit",
        ],
        [
            "MCExploitDecoupled",
        ],
        [
            "MCExploitWeight",
        ],
        [
            "LinearRegression"
        ],
    ]
)

m2 = ExperimentMask(
    "full_vs_weight_all",
    "full optimization vs. weight optimization",
    "full optimization vs. weight optimization by example of algorithms with both implementations",
    [
        [
            "MCExploit",
            "MCOneShot",
            "MCOscillatorAnneal",
            "MCOscillatorAnnealLog",
            "LasVegas",
            "MCExploitAnneal",
        ],
        [
            "MCExploitWeight",
            "MCOneShotWeight",
            "MCOscillatorAnnealWeight",
            "MCOscillatorAnnealLogWeight",
            "LasVegasWeight",
            "MCExploitAnnealWeight",
        ],
        [
            "LinearRegression"
        ],
    ],
)

m3 = ExperimentMask(
    "annealing",
    "implementations of sim. annealing",
    "comparison of simulated annealing inspired algorithms",
    [
        [
            "MCOscillatorAnneal",
        ],
        [
            "MCOscillatorAnnealLog",
        ],
        [
            "MCOscillatorAnnealWeight",
        ],
        [
            "MCOscillatorAnnealLogWeight",
        ],
        [
            "MCExploitAnneal",
        ],
        [
            "MCExploitAnnealWeight",
        ],
        [
            "LinearRegression",
        ]
    ],
)

m4 = ExperimentMask(
    "best_algos",
    "best algorithms",
    "comparison of best full and weight-only optimizing algorithms",
    [   # comparison of best in each family
        [
            "MCExploit", # best monte carlo in oscillator replacement
        ],
        [
            "BasinHopping", # best monte carlo in weight optimization
        ],
        [
            "LinearRegression", # best gradient based algorithm
        ],
    ],
)

m5 = ExperimentMask(
    "best_by_family",
    "best weight-only optimizing algorithms by algorithm family",
    "comparison of weight-only optimizing algorithms by algorithm family",
    [
        [
            "MCExploitNeighborWeight", # best Gibbs sampler
        ],
        [
            "MCOneShotWeight", # brute force search
        ],
        [
            "BasinHopping", # best global and local search algorithm
        ],
        [
            "DifferentialEvolution", # best population based algorithm
        ],
        [
            "LinearRegression", # best gradient based algorithm
        ],
    ],
)

m6 = ExperimentMask(
    "greedy_vs_ergodic",
    "greedy vs. ergodic fully optimizing algorithms",
    "greedy vs ergodic for oscillator and weight optimizing algorithms",
    [
        [
            "MCExploit",
        ],
        [
            "MCExploitErgodic",
        ],
        [
            "MCExploitAnneal",
        ],
    ],
)

algo_masks = [m1, m2, m3, m4, m5, m6]

def main():
    print(m1.get_algo_names())
    print(m1.get_color_map())

if __name__ == "__main__":
    main()