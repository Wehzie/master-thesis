"""
This module implements multiple instances of the ExperimentMask class.
Each instance describes which algorithms are compared against each other.
"""

import mask_type

m1 = mask_type.ExperimentMask(
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

m2 = mask_type.ExperimentMask(
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
            "ScipyAnneal"
        ],
        [
            "ScipyDualAnneal",
        ]
    ],
)

m3 = mask_type.ExperimentMask(
    "best_by_family",
    "best algorithms by family",
    "comparison of best full and weight-only optimizing algorithms",
    [   # comparison of best in each family
        [
            "MCExploit", # best monte carlo in oscillator replacement
        ],
        [
            "MCExploitWeight", # like above but weight only
        ],
        [
            "MCOneShot", # best brute force search
        ],
        [
            "LasVegas", # best Las Vegas algorithm
        ],
        [
            "DifferentialEvolution", # best population based algorithm
        ],
        [
            "BasinHopping", # best monte carlo in weight optimization
        ],
        [
            "LinearRegression", # best gradient based algorithm
        ],
    ],
)

m4 = mask_type.ExperimentMask(
    "best_by_family_weight",
    "best weight-only optimizing algorithms by algorithm family",
    "comparison of weight-only optimizing algorithms by algorithm family",
    [
        [
            "MCExploitWeight", # best Gibbs sampler
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

m5 = mask_type.ExperimentMask(
    "exploit_vs_exploit",
    "MCExploit, j=1 vs. j=10 vs. MCExploitErgodic",
    "comparison of MCExploit with j=1 and j=10 and MCExploitErgodic",
    [
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
        [
            "LinearRegression",
        ]
    ],
)

m6 = mask_type.ExperimentMask(
    "weird_algos",
    "weird algorithms",
    "comparison of weird algorithms against MCExploit and linear regression",
    [
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
        [
            "LinearRegression",
        ]
    ],
)

m7 = mask_type.ExperimentMask(
    "best3_exploit",
    "best three algorithms with mc exploit",
    "comparison of best 3 algorithms",
    [
        [
            "MCExploitWeight",
        ],
        [
            "LasVegas",
        ],
        [
            "LinearRegression",
        ]
    ]
)

m8 = mask_type.ExperimentMask(
    "algo_by_j_replaced_oscillators",
    "simultaneously replaced oscillators",
    "comparison of similar algorithms with varying numbers of simultaneously replaced oscillators",
    [
        [
            "Linear Regression",
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
        ]
    ]
)

# Linear Regression + OneShot + Exploit + Exploit J=10 + Oscillator Annealing in z-ops 0, 1e3, 1e4

algo_masks = [m1, m2, m3, m4, m5, m6, m7, m8]
