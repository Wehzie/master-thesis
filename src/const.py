from typing import Final
import numpy as np
MULTIPROCESSING: Final = False
if MULTIPROCESSING: 
    GLOBAL_SEED = None # TODO: array of seeds and generators, one for each process
else:
    GLOBAL_SEED = 5
RNG: Final = np.random.default_rng(GLOBAL_SEED)
TEST_RNG: Final = np.random.default_rng() # use no seed for testing to increase probability of finding errors
SAMPLE_FLUSH_PERIOD: Final = 1000 # number of samples to store in RAM before flush to disk
LEGAL_DISTS: Final = [RNG.uniform.__name__, RNG.normal.__name__] # supported distributions to draw from
TEST_PARAMS: Final[bool] = True

SEPARATE_LEGEND: Final[bool] = False # show and save legends in a separate figure
MAX_TARGET_DURATION: Final[float] = 10  # maximum duration of a target signal in seconds for frequency sweep
                                        # frequency 0 shouldn't be possible to generate
                                        # a period of T=1/f=10s is already longer than the longest target I plan to generate
                                        # which yields f=1/T=0.1 Hz