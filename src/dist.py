"""
This module implements the distribution class.

A distribution is a random variable that can be sampled.
"""

import numpy as np
import const
from typing import Tuple, Union, Callable


# TODO: replace with something more readable
class Dist:
    """define distributions for random drawing"""

    def __init__(self, dist: Union[int, float, Callable, None], n: int = None, *args, **kwargs):
        if isinstance(dist, Callable):
            # rng1.uniform != rng2.uniform, therefore must use name
            assert dist.__name__ in const.LEGAL_DISTS, "unsupported distribution"

            self.dist = dist
            self.n = n
            self.args = args
            self.kwargs = kwargs
        elif isinstance(dist, int) or isinstance(dist, float):
            self.dist = self.callable_const
            self.n = n
            self.args = (float(dist),)
            self.kwargs = kwargs  # {"const": float(dist)}
            # either args or kwargs could be used here
        elif dist is None:
            self.dist = self.callable_const
            self.n = n
            self.args = (None,)
            self.kwargs = kwargs

    def __repr__(self) -> str:
        if isinstance(self.dist, Callable):
            dist = self.dist.__name__  # e.g. uniform
        else:
            dist = str(self.dist)  # e.g. 42
        dist += " with "
        kwargs = "kwargs=" + str(self.kwargs)
        n = " and n=" + str(self.n)
        return dist + kwargs + n

    def callable_const(self, const, size=None) -> Union[float, np.ndarray]:
        """return a constant value"""
        if size:
            return np.repeat(const, size)
        return const

    def draw(self) -> float:
        """draw a single value from the distribution"""
        return self.dist(*self.args, **self.kwargs)

    def draw_n(self) -> np.ndarray:
        """draw n values from the distribution"""
        return self.dist(*self.args, **self.kwargs, size=self.n)

    def is_uniform(self) -> bool:
        """return True if the distribution is a uniform distribution"""
        return self.dist.__name__ == "uniform"

    def is_normal(self) -> bool:
        """return True if the distribution is a normal distribution"""
        return self.dist.__name__ == "normal"

    def is_const(self) -> bool:
        """return True if the distribution is a constant value"""
        return self.dist.__name__ == "callable_const"

    def get_low_high(self) -> Tuple[float, float]:
        """return the parameters of the distribution"""
        if self.is_uniform():
            return self.kwargs["low"], self.kwargs["high"]
        if self.is_normal():
            return (
                self.kwargs["loc"] - self.kwargs["scale"],
                self.kwargs["loc"] + self.kwargs["scale"],
            )
        if self.is_const():
            return self.draw(), self.draw()
        else:
            raise NotImplementedError

    def compute_range(self) -> None:
        """
        compute the numerical range covered by a distribution.

        for example, a uniform distribution between 0 and 10 has a range of 10.
        """
        if self.is_uniform():
            self.range = self.kwargs["high"] - self.kwargs["low"]
        elif self.is_normal():
            self.range = 2 * self.kwargs["scale"]
        elif self.is_const():
            self.range = 0
        else:
            raise NotImplementedError


class WeightDist(Dist):
    """a distribution for drawing weights"""

    def __init__(self, dist: Union[int, float, Callable], n: int, *args, **kwargs):
        if isinstance(dist, Callable):
            rng = np.random.default_rng()
            assert dist.__name__ in [
                rng.uniform.__name__,
                rng.normal.__name__,
            ], "unsupported distribution"
            self.dist = dist
            self.n = n  # draw n instead of 1, for weight drawing
            self.args = args
            self.kwargs = kwargs
        elif isinstance(dist, int | float):
            self.dist = self.callable_const
            self.n = n
            self.args = (float(dist),)
            self.kwargs = kwargs


if __name__ == "__main__":
    dist = Dist(None)
    print(dist.draw())
