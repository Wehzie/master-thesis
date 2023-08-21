"""
This module implements Las Vegas algorithms for optimization.

Las vegas algorithms have non-deterministic runtime and draw from randomness to find the best solution to optimization.
"""

import copy
import gen_signal_python as gensi_python  # TODO: remove once init_las_vegas is removed
import algo
import sample
import const

from typing import List, Tuple, Union

import numpy as np


class LasVegas(algo.SearchAlgo):
    """Begins with an empty ensemble and only adds an oscillator to the ensemble if RMSE is reduced."""

    def init_las_vegas(self, weight_init: Union[None, str], store_det_args: bool) -> sample.Sample:
        """initialize the best sample by different methods"""
        # TODO: remove this function
        # possibly reuse different initialization methods
        if weight_init is None:  # in this mode, weights are not adapted
            signal_matrix = np.zeros((self.rand_args.n_osc, self.rand_args.samples))
            signal_args = list()
        else:
            signal_matrix, signal_args = gensi_python.PythonSigGen.draw_n_oscillators(
                self.rand_args, store_det_args
            )

        if weight_init == "ones":
            weights = np.ones(self.rand_args.n_osc)
        elif weight_init == "uniform":
            weights = const.RNG.uniform(-1, 1, self.rand_args.n_osc)
        elif weight_init == "zeros":
            weights = np.zeros(self.rand_args.n_osc)
        elif weight_init == "dist":
            dist = copy.deepcopy(self.rand_args.weight_dist)
            dist.kwargs["size"] = self.rand_args.n_osc
            weights = dist.draw()
        else:
            weights = None  # in this mode, weights are not adapted

        return sample.Sample(
            signal_matrix, weights, np.sum(signal_matrix, axis=0), 0, np.inf, signal_args
        )

    def infer_k_from_z(self) -> int:
        """infer the number of iterations k from the number of allowed perturbations z"""
        return None

    def init_best_sample(self) -> sample.Sample:
        """initialize the best sample"""
        return self.gen_zero_sample()

    def draw_temp_sample(
        self, base_sample: sample.Sample, osc_to_replace: List[int]
    ) -> sample.Sample:
        """draw a temporary sample from the base sample by replacing a subset of oscillators"""
        return self.draw_partial_sample(base_sample, osc_to_replace)

    def comp_samples(
        self, base_sample: sample.Sample, temp_sample: sample.Sample
    ) -> Tuple[sample.Sample, bool]:
        """
        compare two samples and return the one with lower rmse.

        returns:
            sample: the sample with lower rmse
            changed: true if the new sample is better than the old one, false if the old one is better
        """
        if temp_sample.rmse < base_sample.rmse:
            return temp_sample, True
        return base_sample, False

    def search(self, *args, **kwargs) -> Tuple[sample.Sample, int]:
        """
        generate k-signals which are a sum of n-oscillators.

        model is constructed by aggregation
        oscillator candidates are accepted into the model when they lower RMSE

        returns:
            best_sample: the best sample found
            z_ops: the number of operations performed
        """
        print(f"searching with {self.__class__.__name__}")
        self.clear_state()
        self.handle_mp(kwargs)

        def generator():  # support tqdm timer and iterations per second
            while not self.stop_on_z_ops():
                yield

        best_sample = self.init_best_sample()
        for k in generator():
            base_sample = self.init_best_sample()
            i = 0  # number of replaced weights
            while i < self.rand_args.n_osc + 1 and not self.stop_on_z_ops():  # +1 for offset
                temp_sample = self.draw_temp_sample(base_sample, np.array([i]))
                base_sample, changed = self.comp_samples(base_sample, temp_sample)
                if changed:
                    i += 1  # move to next row

            self.manage_state(base_sample, k)
            best_sample, _ = self.comp_samples(best_sample, base_sample)

        return best_sample, self.z_ops


class LasVegasWeight(LasVegas):
    """Begins with a random ensemble."""

    def init_best_sample(self) -> sample.Sample:
        """initialize the best sample"""
        return self.draw_sample()

    def draw_temp_sample(
        self, base_sample: sample.Sample, osc_to_replace: List[int]
    ) -> sample.Sample:
        """draw a temporary sample from the base sample by replacing a subset of oscillators"""
        return self.draw_partial_sample_weights(base_sample, osc_to_replace)
