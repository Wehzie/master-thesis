"""
This module implements evolutionary optimization algorithms that rely on a population of solution candidates.

In a neuromorphic context evolutionary algorithms are more feasible than gradient based methods,
but they are harder to implement than monte carlo methods, as candidate solutions need to be tracked.
"""

from scipy.optimize import differential_evolution
import numpy as np
import algo
import sample
import data_analysis
import const


class DifferentialEvolution(algo.SearchAlgo):
    """implement differential evolution algorithm using scipy.optimize.differential_evolution"""

    def infer_k_from_z(self) -> int:
        """infer the number of loops/iterations k from the maximum number of perturbations z"""
        return None

    def init_best_sample(self) -> sample.Sample:
        """initialize the best sample with random oscillators"""
        return self.draw_sample()

    def search(self, *args, **kwargs):
        """search for the best ensemble of oscillators"""
        print(f"searching with {self.__class__.__name__}")

        def eval_func(weights, *args):
            # offset is last element in weights
            signal_matrix, target = args
            weighted_sum = np.sum(signal_matrix.T * weights[:-1], axis=1) + weights[-1]
            return data_analysis.compute_rmse(weighted_sum, target)

        self.clear_state()
        self.handle_mp(kwargs)

        # pack model and data into args
        best_sample = self.init_best_sample()
        args = (best_sample.signal_matrix, self.target)

        # search bounds
        lo, hi = self.rand_args.weight_dist.get_low_high()
        weight_bounds = [(lo, hi) for _ in range(self.rand_args.n_osc)]
        lo, hi = self.rand_args.offset_dist.get_low_high()
        offset_bounds = (lo, hi)
        weight_offset_bounds = weight_bounds + [offset_bounds]

        # compute number of generations from max_z_ops
        num_populations = 15  # scipy default
        oscillators_per_generation = num_populations * self.rand_args.n_osc
        num_generations = int((self.max_z_ops - self.rand_args.n_osc) // oscillators_per_generation)

        # run search
        result = differential_evolution(
            eval_func,
            weight_offset_bounds,
            args=args,
            maxiter=num_generations,
            seed=const.GLOBAL_SEED,
        )

        # update best_sample and z_ops
        best_sample.weights = result.x[:-1]
        best_sample.offset = result.x[-1]  # offset is last element
        best_sample.update(self.target)
        self.z_ops += num_generations * oscillators_per_generation
        return best_sample, self.z_ops
