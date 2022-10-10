import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import numpy as np

import param_types as party

def test_lazy():
    rng = np.random.default_rng() # no seed needed since not used to draw
    res = party.Dist(rng.uniform, low=0, high=1e4)
    assert isinstance(res.draw(), float)

    res = party.Dist(rng.uniform, low=0, high=1e4, n = 5)
    assert len(res.draw_n()) == 5

    res = party.Dist(1)
    assert isinstance(res.draw(), float)

    res = party.Dist(1, n = 5)
    assert len(res.draw_n()) == 5