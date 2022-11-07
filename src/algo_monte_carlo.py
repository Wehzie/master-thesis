"""
monte carlo algorithms have deterministic runtime and draw from randomness to find the best solution.

monte carlo algorithms are easiest to implement in hardware;
no gradient is propagated nor must complex memory be maintained.
"""

from typing import Tuple

import algo
import sample
import const
import data_analysis

import numpy as np
from tqdm import tqdm
from scipy.optimize import basinhopping

class MonteCarlo(algo.SearchAlgo):
    """abstract class"""

    def search(self, *args, **kwargs) -> Tuple[sample.Sample, int]:
        """generate k-signals which are a sum of n-oscillators
        on each iteration draw a new full model (matrix of n-oscillators)
        """
        self.clear_state()
        self.handle_mp(kwargs)

        best_sample = self.init_best_sample()
        for k in tqdm(range(self.k_samples)):
            temp_sample = self.draw_temp_sample(best_sample, k)
            best_sample = self.comp_samples(best_sample, temp_sample)
            self.manage_state(temp_sample, k)

        return best_sample, self.z_ops



class MCOneShot(MonteCarlo):
    """monte carlo algorithm for samples consisting of independent oscillators"""

    def infer_k_from_z(self) -> int:
        # cost of initializing best_sample is zero
        z_loop = self.rand_args.n_osc * 2   # draw a new sample with n oscillators and weights on each loop
        return int(self.max_z_ops // z_loop)

    def init_best_sample(self) -> sample.Sample:
        return self.gen_empty_sample()

    def draw_temp_sample(self, *args, **kwargs) -> sample.Sample:
        return self.draw_sample()

class MCOneShotWeight(MCOneShot):
    """Use the MCOneShot algorithm but only draw new weights for each sample"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 2   # initialize a best sample with n oscillators and weights
        z_loop = self.rand_args.n_osc       # draw a sample with n new weights each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        return self.draw_sample_weights(base_sample)



class MCExploit(MonteCarlo):
    """monte carlo algorithm exploiting a single sample by iterative re-draws
    
    self.mp must be set to True or False"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 2 # initialize a best sample with n oscillators and weights
        z_loop = self.j_replace * 2 # j weights and oscillators updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_sample(base_sample, osc_to_replace)

class MCExploitWeight(MCExploit):
    """Use the MCExploit algorithm but only draw new weights for each sample"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 2
        z_loop = self.j_replace # j weights are updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_sample_weights(base_sample, osc_to_replace)



class MCAnneal(MonteCarlo):
    """use a schedule to reduce the number of oscillators across iterations.
    akin to simulated annealing"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 2
        z_loop = self.rand_args.n_osc
        # we have a linear schedule in the range (n, 1)
        # the total number of operations is then sum(range(n, 1))
        # the average number of operations is sum(range(1, n)) / n
        # that's the same as n / 2 because the schedule is linear
        # because oscillators and weights are updated we add * 2
        # the result is z_loop = n
        return int((self.max_z_ops - z_init) // z_loop)
    
    def read_j_from_schedule(self, k: int) -> int:
        """return the number of oscillator or weights to draw according to a linear schedule
        
        params:
            k: iteration number
            
        returns:
            j: number of oscillators or weights to draw"""
        temperature = (1 - k / self.k_samples)
        return np.ceil(self.rand_args.n_osc*temperature).astype(int)

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, k: int, *args, **kwargs) -> sample.Sample:
        j_replace = self.read_j_from_schedule(k)
        osc_to_replace = self.draw_random_indices(j_replace)
        return self.draw_partial_sample(base_sample, osc_to_replace)

class MCAnnealWeight(MCAnneal):
    """use the MCAnneal algorithm but only draw new weights for each sample"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 2
        z_loop = self.rand_args.n_osc // 2 # we only draw new weights and omit * 2 for oscillators
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, k: int, *args, **kwargs) -> sample.Sample:
        j_replace = self.read_j_from_schedule(k)
        osc_to_replace = self.draw_random_indices(j_replace)
        return self.draw_partial_sample_weights(base_sample, osc_to_replace)

class MCAnnealLog():
    """use logarithmic schedule to reduce the number of oscillators across iterations."""

class MCPurge():
    """
    initialize pool of n oscillators and weights
    set a weight to 0 and evaluate the sample
    accept change if rmse is lower
    stop after looping over each oscillator or when z_ops is exhausted
    """

class MCGrowShrink():
    """
    initialize pool of n oscillators and weights
    randomly add or remove an oscillator
    where add means draw and replace
    where remove means set weight to 0
    evaluate the sample
    accept change if rmse is lower
    stop after looping over each oscillator or when z_ops is exhausted
    """

class BasinHopping(algo.SearchAlgo):
    """NOTE: this algorithm uses gradient information when used with 
        scipy.optimize.minimize("method": "BFGS") and some other methods,
        these are not interesting for our neuromorphic context"""

    def draw_temp_sample(self) -> sample.Sample:
        return super().draw_temp_sample()
    
    def infer_k_from_z(self) -> int:
        return None

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()
        
    def search(self, minimizer_kwargs={"method": "COBYLA"}, *args, **kwargs):
        """
        
        params:
            minimizer_kwargs: dict of arguments to pass to scipy.optimize.minimize
                                note that only non-gradient based algorithms should be used.
                                a non gradient based algorithm is the Constrained Optimization BY Linear Approximation (COBYLA) algorithm.
        """

        self.clear_state()
        self.handle_mp(kwargs)

        def eval_func(weights):
            """adapt weights for simulated annealing"""
            weighted_sum = np.sum(best_sample.signal_matrix.T * weights, axis=1)
            return data_analysis.compute_rmse(weighted_sum, self.target)

        best_sample = self.init_best_sample()
        
        # weight bounds
        lo = self.rand_args.weight_dist.kwargs["low"]
        hi = self.rand_args.weight_dist.kwargs["high"]
        
        # define bounds
        class Bounds:
            def __init__(self, low: float, high: float):
                self.low = low
                self.high = high
            def __call__(self, **kwargs):
                x = kwargs["x_new"] # hard coded kwarg in scipy
                tmin = bool(np.all(x >= self.low))
                tmax = bool(np.all(x <= self.high))
                return tmax and tmin
        lo = self.rand_args.weight_dist.kwargs["low"]
        hi = self.rand_args.weight_dist.kwargs["high"]
        weight_bounds = Bounds(lo, hi)

        niter = int(self.max_z_ops // self.rand_args.n_osc)

        result = basinhopping(eval_func, best_sample.weights, minimizer_kwargs=minimizer_kwargs, niter=niter, accept_test=weight_bounds, seed=const.GLOBAL_SEED)

        best_sample.weights = result.x
        best_sample.update(self.target)
        self.z_ops += niter * self.rand_args.n_osc
        
        return best_sample, self.z_ops

