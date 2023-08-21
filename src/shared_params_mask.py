"""
This module implements multiple instances of the ExperimentMask class.

Each instance describes which algorithms are compared against each other.
"""

import mask_type

m1 = mask_type.ExperimentMask(
    "full_vs_weight",
    "full optimization vs. weight optimization",
    "full optimization vs. weight optimization by example of algorithms with both implementations",
    [
        ["LinearRegression"],
        [
            "MCExploit",
        ],
        [
            "MCOneShot",
        ],
        [
            "MCOscillatorAnneal",
        ],
        [
            "MCOscillatorAnnealLog",
        ],
        [
            "LasVegas",
        ],
        [
            "MCExploitAnneal",
        ],
        ### weight only
        [
            "MCExploitWeight",
        ],
        [
            "MCOneShotWeight",
        ],
        [
            "MCOscillatorAnnealWeight",
        ],
        [
            "MCOscillatorAnnealLogWeight",
        ],
        [
            "LasVegasWeight",
        ],
        [
            "MCExploitAnnealWeight",
        ],
    ],
)

m2 = mask_type.ExperimentMask(
    "annealing",
    "implementations of sim. annealing",
    "comparison of simulated annealing inspired algorithms",
    [
        [
            "LinearRegression",
        ],
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
        ["ScipyAnneal"],
        [
            "ScipyDualAnneal",
        ],
        [
            "BasinHopping",
        ],
    ],
)

m3 = mask_type.ExperimentMask(
    "best_by_family",
    "best algorithms by family",
    "comparison of best full and weight-only optimizing algorithms",
    [  # comparison of best in each family
        [
            "LinearRegression",  # best gradient based algorithm
        ],
        [
            "MCExploit",  # best monte carlo in oscillator replacement
        ],
        [
            "MCExploitWeight",  # like above but weight only
        ],
        [
            "MCOneShot",  # best brute force search
        ],
        [
            "LasVegas",  # best Las Vegas algorithm
        ],
        [
            "DifferentialEvolution",  # best population based algorithm
        ],
        [
            "BasinHopping",  # best monte carlo in weight optimization
        ],
    ],
)

m4 = mask_type.ExperimentMask(
    "best_by_family_weight",
    "best weight-only optimizing algorithms by algorithm family",
    "comparison of weight-only optimizing algorithms by algorithm family",
    [
        [
            "LinearRegression",  # best gradient based algorithm
        ],
        [
            "MCExploitWeight",  # best Gibbs sampler
        ],
        [
            "MCOneShotWeight",  # brute force search
        ],
        [
            "BasinHopping",  # best global and local search algorithm
        ],
        [
            "DifferentialEvolution",  # best population based algorithm
        ],
    ],
)

m5 = mask_type.ExperimentMask(
    "exploit_vs_exploit",
    "MCExploit, j=1 vs. j=10 vs. MCExploitErgodic",
    "comparison of MCExploit with j=1 and j=10 and MCExploitErgodic",
    [
        [
            "LinearRegression",
        ],
        [
            "MCExploit",
        ],
        [
            "MCExploitJ10",
        ],
        [
            "MCExploitErgodic",
        ],
        [
            "MCExploitDecoupled",
        ],
        [
            "MCExploitWeight",
        ],
    ],
)

m6 = mask_type.ExperimentMask(
    "weird_algos",
    "weird algorithms",
    "comparison of weird algorithms against MCExploit and linear regression",
    [
        [
            "LinearRegression",
        ],
        [
            "MCExploit",
        ],
        [
            "MCExploitNeighborWeight",
        ],
        [
            "MCExploitFast",
        ],
        [
            "MCGrowShrink",
        ],
        [
            "MCDampen",
        ],
        [
            "MCPurge",
        ],
    ],
)

m7 = mask_type.ExperimentMask(
    "best3_exploit",
    "best three algorithms with mc exploit",
    "comparison of best 3 algorithms",
    [
        [
            "LinearRegression",
        ],
        [
            "MCExploitWeight",
        ],
        [
            "LasVegas",
        ],
    ],
)

m8 = mask_type.ExperimentMask(
    "algo_by_j_replaced_oscillators",
    "simultaneously replaced oscillators",
    "comparison of similar algorithms with varying numbers of simultaneously replaced oscillators",
    [
        [
            "LinearRegression",
        ],
        [
            "MCOneShot",
        ],
        [
            "MCExploit",
        ],
        [
            "MCExploitJ10",
        ],
        [
            "MCOscillatorAnneal",
        ],
        [
            "MCOscillatorAnnealLog",
        ],
        # [ # include ergodic here to reduce number of figures
        #     "MCExploitErgodic",
        # ],
    ],
)

m9 = mask_type.ExperimentMask(
    "full_vs_weight_focus",
    "focus on full optimization vs. weight optimization",
    "full optimization vs. weight optimization with some algorithms",
    [
        ### benchmark
        ["LinearRegression"],
        [
            "MCOneShot",
        ],
        [
            "MCExploit",
        ],
        [
            "LasVegas",
        ],
        ### weight only
        [
            "MCExploitWeight",
        ],
        [
            "MCOneShotWeight",
        ],
        [
            "LasVegasWeight",
        ],
        ### mixed
        [
            "MCExploitDecoupled",
        ],
    ],
)

m10 = mask_type.ExperimentMask(
    "focus_exploits",
    "normal vs ergodic vs decoupled",
    "comparison of MCExploit with j=1 and j=10 and MCExploitErgodic",
    [
        [
            "LinearRegression",
        ],
        ["MCExploit"],
        [
            "MCExploitWeight",
        ],
        [
            "MCExploitErgodic",
        ],
        [
            "MCExploitNeighborWeight",
        ],
    ],
)


algo_masks = [m2, m8, m9, m10]
