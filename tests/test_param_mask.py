
# add code in src to path
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import param_mask
import param_util

algo_names = [Algo(None).__class__.__name__ for Algo in param_util.algo_list] # algo names as strings

def test_lists_match():
    for mask in param_mask.algo_masks:
        for cluster in mask:
            for algo in cluster:
                assert algo in param_util.algo_names, "unknown algorithm in algo_masks"