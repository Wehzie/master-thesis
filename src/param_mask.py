"""
set of masks over the experiment results
each mask contains clusters of algorithms which are compared
"""

algo_masks = (
[ # set of masks
    [ # mask
        [ # cluster
            "MCExploit", # algorithm in a cluster
        ],
        # this cluster compares
        # weight vs no weight optimization by example of MCExploit
        [
            "MCExploitDecoupled",
        ],
        [
            "MCExploitWeight",
        ],
        [
            "LinearRegression"
        ],
    ],
    [   # comparing weight vs no weight in algorithms with clear equivalent
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
    [   # comparing types of annealing
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
    [   # comparison of best in each family
        [
            "MCExploit", # best monte carlo in oscillator replacement
        ],
        [
            "BasinHopping", # best monte carlo in weight optimization
        ],
        [
            "DifferentialEvolution", # best population based algorithm
        ],
        [
            "LinearRegression", # best gradient based algorithm
        ],
    ],
    [   # comparison of best in each algorithm class for weight only optimization
        [
            "MCExploitNeighborWeight", # best gibbs sampler
        ],
        [
            "MCOneShotWeight", # brute force search
        ],
        [
            "BasinHopping", # best non gradient based weight optimizer
        ],
        [
            "DifferentialEvolution", # best population based algorithm
        ],
        [
            "LinearRegression", # best gradient based algorithm
        ],
    ],
    [   # greedy vs ergodic for oscillator and weight searching algorithms
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
]
)

