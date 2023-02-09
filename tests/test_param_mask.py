
# add code in src to path
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import param_mask
import parameter_builder

algo_names = [Algo(None).__class__.__name__ for Algo in parameter_builder.algo_list] # algo names as strings

def test_lists_match():
    for mask in param_mask.algo_masks:
        for algo in mask.get_algo_names():
            assert algo in algo_names, "unknown algorithm in algo_masks"