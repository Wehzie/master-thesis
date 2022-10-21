import numpy as np
import const
from typing import List, Union, Callable
from util import add_str2keys

# "|", for example int|float requires python 3.10 or greater
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
        if "attr_name" in self.kwargs:
            attr_name = self.kwargs["attr_name"] # attribute relate to the distribution, e.g. frequency
        else:
            attr_name = "yolo" # TODO: for testing only

        if isinstance(self.dist, Callable):
            dist = self.dist.__name__        # e.g. uniform
        else:
            dist = str(self.dist)            # e.g. 42

        kwargs = add_str2keys(attr_name, self.kwargs)

        return kwargs

    # def __repr__(self) -> str:
    #     if isinstance(self.dist, Callable):
    #         dist = self.dist.__name__
    #     else:
    #         dist = str(self.dist)
    #     dist += ", "
    #     args = str(self.args) + ", "
    #     kwargs = str(self.kwargs)
    #     return dist + args + kwargs + "\n"
    


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