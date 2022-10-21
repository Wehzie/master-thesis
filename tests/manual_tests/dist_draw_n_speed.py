"""this snippet compares to forms of drawing n weights form a dist object for speed

the numpy form is clearly faster
"""

import data_analysis # @print_time
from typing import Union, Callable
import numpy as np
import const

class Dist:
    """define distributions for random drawing"""
    def __init__(self, dist: Union[int, float, Callable], n: int = None, *args, **kwargs):
        if isinstance(dist, Callable):
            rng = np.random.default_rng() # no seed needed since not used to draw
            # rng1.uniform != rng2.uniform, therefore must use name
            assert dist.__name__ in const.LEGAL_DISTS, "unsupported distribution"
            
            self.dist = dist
            self.n = n
            self.args = args
            self.kwargs = kwargs
        elif isinstance(dist, int|float):
            self.dist = self.callable_const
            self.n = n
            self.args = (float(dist),)
            self.kwargs = kwargs  # {"const": float(dist)}
                                    # either args or kwargs could be used here

    def callable_const(self, const, size=None) -> Union[float, np.ndarray]:
        if size: return np.repeat(const, size)
        return const

    def draw(self) -> float:
        return self.dist(*self.args, **self.kwargs)
    
    def draw_n(self) -> np.ndarray:
        return self.dist(*self.args, **self.kwargs, size=self.n)
    
    def __repr__(self) -> str:
        if isinstance(self.dist, Callable):
            dist = self.dist.__name__
        else:
            dist = str(self.dist)
        dist += ", "
        args = str(self.args) + ", "
        kwargs = str(self.kwargs)
        return dist + args + kwargs + "\n"

class WeightDist(Dist):
    def __init__(self, dist: Union[int, float, Callable], n: int, *args, **kwargs):
        if isinstance(dist, Callable):
            rng = np.random.default_rng()
            assert dist.__name__ in [rng.uniform.__name__, rng.normal.__name__], "unsupported distribution"
            self.dist = dist
            self.n = n                  # draw n instead of 1, for weight drawing
            self.args = args
            self.kwargs = kwargs
        elif isinstance(dist, int|float):
            self.dist = self.callable_const
            self.n = n
            self.args = (float(dist),)
            self.kwargs = kwargs

@data_analysis.print_time
def case_a(n: int):
    d = WeightDist(const.RNG.uniform, low=1, high=10, n=n)
    return d.draw_n()

@data_analysis.print_time
def case_b(n: int):
    d = Dist(const.RNG.uniform, low=1, high=10)
    return np.array([d.draw() for _ in range(n)])

a = case_a(10000)
b = case_b(10000)
