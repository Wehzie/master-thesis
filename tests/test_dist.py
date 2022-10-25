import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import dist
import const

rng = const.RNG

def test_lazy():

    # Dist

    # draw from uniform distribution
    d = dist.Dist(rng.uniform, low=0, high=1e4)
    assert isinstance(d.draw(), float) or isinstance(d.draw(), int)

    d = dist.Dist(rng.uniform, low=0, high=1e4, n = 5)
    assert len(d.draw_n()) == 5

    # draw from normal distribution
    d = dist.Dist(rng.normal, loc=0, scale=10)
    assert isinstance(d.draw(), float) or isinstance(d.draw(), int)

    d = dist.Dist(rng.normal, loc=0, scale=10, n = 5)
    assert len(d.draw_n()) == 5

    # draw constant
    d = dist.Dist(1)
    assert isinstance(d.draw(), float) or isinstance(d.draw(), int)

    d = dist.Dist(1, n = 5)
    assert len(d.draw_n()) == 5

    # WeightDist

    # draw from uniform distribution
    d = dist.WeightDist(rng.uniform, low=0, high=1e4, n = 5)
    assert isinstance(d.draw(), float) or isinstance(d.draw(), int)
    assert len(d.draw_n()) == 5

    # draw from normal distribution
    d = dist.WeightDist(rng.normal, loc=0, scale=10,n = 5)
    assert isinstance(d.draw(), float) or isinstance(d.draw(), int)
    assert len(d.draw_n()) == 5

    # draw constant
    d = dist.WeightDist(1, n = 5)
    assert isinstance(d.draw(), float) or isinstance(d.draw(), int)
    assert len(d.draw_n()) == 5