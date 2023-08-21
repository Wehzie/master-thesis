"""
This module implements Markov Chain Monte Carlo optimization algorithms.

All algorithms have a non-zero probability of accepting worse candidate solutions.
"""
import algo_monte_carlo
import sample
import const
import data_analysis
import algo
import algo_args_type as algarty

import numpy as np
from scipy.optimize import basinhopping, dual_annealing


class MCExploitErgodic(algo_monte_carlo.MCExploit):
    """
    An ergodic version of the MCExploit algorithm.

    Compared to MCExploit, this algorithm accepts worse samples with a small probability.
    The probability remains constant throughout runtime, opposed to MCExploitAnneal.
    """

    def accept_candidate_sample(
        self, rmse_base: float, rmse_temp: float, temperature: float = 0.1
    ) -> bool:
        """
        Accept a new sample following some rules.

        Acceptance rules are as follows:
            rmse_temp < rmse_base --> 100% acceptance
            rmse_temp == rmse_base --> 50% acceptance when temperature=1
            rmse_temp == infinity --> 0% acceptance
        """
        if rmse_temp < rmse_base:
            return True
        inv_temp = 1 / temperature
        acceptance_ratio = 1 / (1 + rmse_temp / rmse_base) ** inv_temp
        return const.RNG.uniform(0, 1) <= acceptance_ratio


class MCExploitAnneal(algo_monte_carlo.MCExploit):
    """
    An ergodic version of the MCExploit algorithm.

    Compared to MCExploit, this algorithm accepts worse samples with a small probability
    The probability to accept worse candidates decreases over time, as the temperature decreases.
    """

    def get_temp(self, k: int) -> float:
        """compute temperature for k-th iteration"""
        return 1 - (k + 1) / self.k_samples

    def accept_candidate_sample(
        self, base_rmse: float, temporary_rmse: float, temperature: float
    ) -> bool:
        """define the acceptance function follow the definition of Kirkpatrick et al. (1983)"""
        if temporary_rmse < base_rmse:
            return True
        acceptance_ratio = np.exp(-(temporary_rmse - base_rmse) / temperature)
        return const.RNG.uniform(0, 1) <= acceptance_ratio

    def comp_samples(
        self, base_sample: sample.Sample, temp_sample: sample.Sample, iteration_k: int
    ) -> sample.Sample:
        """compare two oscillator ensembles and return a winner based on the acceptance function"""
        temperature = self.get_temp(iteration_k)
        if self.accept_candidate_sample(base_sample.rmse, temp_sample.rmse, temperature):
            return temp_sample
        return base_sample


class MCExploitAnnealWeight(MCExploitAnneal):
    """Weight only optimizing version of MCExploitErgodicAnneal."""

    def infer_k_from_z(self) -> int:
        """infer the number of iterations k from a maximum number of perturbations z"""
        z_init = (
            self.rand_args.n_osc * 3 + 1
        )  # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        z_loop = self.j_replace  # j weights are updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """draw a temporary sample, a neighbor of the base sample"""
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_sample_weights(base_sample, osc_to_replace)


class BasinHopping(algo.SearchAlgo):
    """implement the Basin Hopping algorithm in scipy.optimize.basinhopping.

    NOTE: this algorithm uses gradient information when used with
        scipy.optimize.minimize("method": "BFGS") and some other methods,
        these are not interesting for our neuromorphic context,
        we use gradient free COBYLA (Constrained Optimization BY Linear Approximation)
    """

    def infer_k_from_z(self) -> int:
        """ignore iterations k and use `maxiter` instead, see search()"""
        return None

    def init_best_sample(self) -> sample.Sample:
        """initialize the best sample by drawing a random sample"""
        return self.draw_sample()

    def search(self, *args, **kwargs):
        """
        optimize the oscillator ensemble by basin hopping.

        params:
            minimizer_kwargs: dict of arguments to pass to scipy.optimize.minimize
                                note that only non-gradient based algorithms should be used.
                                a non gradient based algorithm is the Constrained Optimization BY Linear Approximation (COBYLA) algorithm.
        """
        print(f"searching with {self.__class__.__name__}")

        self.clear_state()
        self.handle_mp(kwargs)

        best_sample = self.init_best_sample()

        def eval_func_offset(offset):
            """adapt offset by simulated annealing"""
            weighted_sum = (
                np.sum(best_sample.signal_matrix.T * best_sample.weights, axis=1) + offset
            )
            return data_analysis.compute_rmse(weighted_sum, self.target)

        def eval_func_weight(weights):
            """adapt weights for simulated annealing"""
            weighted_sum = (
                np.sum(best_sample.signal_matrix.T * weights, axis=1) + best_sample.offset
            )
            return data_analysis.compute_rmse(weighted_sum, self.target)

        # define bounds
        class Bounds:
            def __init__(self, low: float, high: float):
                self.low = low
                self.high = high

            def __call__(self, **kwargs):
                x = kwargs["x_new"]  # hard coded kwarg in scipy
                tmin = bool(np.all(x >= self.low))
                tmax = bool(np.all(x <= self.high))
                return tmax and tmin

        # optimize the offset before optimizing the weights since SciPy's Bounds class is too restrictive to optimize both simultaneously
        # the documentation may also just be too confusing at explaining how to do this
        lo, hi = self.rand_args.offset_dist.get_low_high()
        offset_bounds = Bounds(lo, hi)
        niter = int(self.max_z_ops // self.rand_args.n_osc) // 20
        result = basinhopping(
            eval_func_offset,
            best_sample.offset,
            minimizer_kwargs={"method": "COBYLA"},
            niter=niter,
            accept_test=offset_bounds,
            seed=const.GLOBAL_SEED,
        )
        best_sample.offset = result.x

        # optimize weights
        lo, hi = self.rand_args.weight_dist.get_low_high()
        weight_bounds = Bounds(lo, hi)
        # using a factor of 1/10, the runtime of the algorithm is similar to that of other algorithms
        niter = int(self.max_z_ops // self.rand_args.n_osc) // 10
        result = basinhopping(
            eval_func_weight,
            best_sample.weights,
            minimizer_kwargs={"method": "COBYLA"},
            niter=niter,
            accept_test=weight_bounds,
            seed=const.GLOBAL_SEED,
        )
        best_sample.weights = result.x

        best_sample.update(self.target)
        self.z_ops += niter * self.rand_args.n_osc

        return best_sample, self.z_ops


class ScipyAnneal(algo.SearchAlgo):
    """
    Generalized Simulated Annealing combines Classical Simulated Annealing (CSA) with Fast Simulated Annealing (FSA).

    reference:
        https://journal.r-project.org/archive/2013/RJ-2013-002/RJ-2013-002.pdf
    """

    def __init__(self, algo_args: algarty.AlgoArgs):
        super().__init__(algo_args)
        self.no_local_search = True

    def infer_k_from_z(self) -> int:
        """ignore iterations k and use `maxfun` instead, see search()"""
        return None

    def init_best_sample(self) -> sample.Sample:
        """initialize the best oscillator ensemble with a random sample"""
        return self.draw_sample()

    def search(self, *args, **kwargs):
        """find the best oscillator ensemble by simulated annealing"""
        print(f"searching with {self.__class__.__name__}")
        self.clear_state()
        self.handle_mp(kwargs)
        best_sample = self.init_best_sample()

        lo, hi = self.rand_args.weight_dist.get_low_high()
        weight_bounds = list(zip([lo] * self.rand_args.n_osc, [hi] * self.rand_args.n_osc))

        def eval_func_weight(weights):
            """adapt weights for simulated annealing"""
            weighted_sum = (
                np.sum(best_sample.signal_matrix.T * weights, axis=1) + best_sample.offset
            )
            return data_analysis.compute_rmse(weighted_sum, self.target)

        maxfun = int(self.max_z_ops // self.rand_args.n_osc)

        result = dual_annealing(
            eval_func_weight,
            bounds=weight_bounds,
            no_local_search=self.no_local_search,
            maxfun=maxfun,
            seed=const.GLOBAL_SEED,
            minimizer_kwargs={"method": "L-BFGS-B"},
        )
        # NOTE: COBYLA preferred as gradient-less method
        # L-BFGS-B kept for integrity of the experiments
        best_sample.weights = result.x

        best_sample.update(self.target)
        self.z_ops += maxfun * self.rand_args.n_osc

        return best_sample, self.z_ops


class ScipyDualAnneal(ScipyAnneal):
    """Generalized Simulated Annealing (GSA) plus a local search after each iteration."""

    def __init__(self, algo_args: algarty.AlgoArgs):
        super().__init__(algo_args)
        self.no_local_search = False
