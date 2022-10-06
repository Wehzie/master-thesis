import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from param_types import PythonSignalDetArgs, PythonSignalRandArgs
from param_types import Dist
import params

import numpy as np

rng = np.random.default_rng(params.GLOBAL_SEED)

k_samples = 3
j_exploits = 100
weight_inits = ["zeros", "ones", "uniform", "dist"]
py_rand_args_uniform = PythonSignalRandArgs(
    n_osc = 100,
    duration = None,
    samples = 300,
    f_dist = Dist(rng.uniform, low=1e5, high=1e6),
    amplitude = 0.5,                                    # resembling 0.5 V amplitude of V02
    weight_dist = Dist(rng.uniform, low=0.1, high=100),   # resistor doesn't amplify so not > 1
    phase_dist = Dist(rng.uniform, low=-1/3, high=1/3), # uniform 0 to 2 pi phase shift seems too wild
    offset_dist = Dist(rng.uniform, low=0, high=0),    # offset should be reasonable and bounded by amplitude*weight
    sampling_rate = 11025                               # the sampling rate of the Magpie signal
)