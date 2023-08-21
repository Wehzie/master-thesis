"""This module implements gradient based optimization algorithms.

In a neuromorphic context gradient based algorithms are difficult because obtaining gradient information is difficult.
This is in particular due to the attribution problem.
This is to ask by what amount does oscillator i contribute to the observed loss function?
"""

from typing import Tuple

import sample
import algo


class LinearRegression(algo.SearchAlgo):
    """implement ordinary least squares regression"""

    def infer_k_from_z(self) -> None:
        """it's not feasible to infer k from z using the scipy implementation of linear regression"""
        return None

    def search(self, *args, **kwargs) -> Tuple[sample.Sample, int]:
        """randomly draw n-oscillators, then apply ordinary least squares linear regression to fit against the target"""
        print(f"searching with {self.__class__.__name__}")
        self.clear_state()
        self.handle_mp(kwargs)
        best_sample = self.draw_sample()
        reg_sample = sample.Sample.regress_sample(best_sample, self.target)

        return reg_sample, self.z_ops
