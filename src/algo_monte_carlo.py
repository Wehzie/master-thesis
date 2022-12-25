"""
monte carlo algorithms have deterministic runtime and draw from randomness to find the best solution.

monte carlo algorithms are easiest to implement in hardware;
no gradient is propagated nor must complex memory be maintained.
"""

import copy
from typing import Tuple, List

import algo
import sample
import const
import data_analysis
import algo_args_types as algarty

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
    """This algorithms most closely resembles brute-force search.
    Compared to brute-force search, this algorithm remembers it's best solution to date.
    The to-date best solution is returned when allocated resources (z-ops) are exhausted."""

    def infer_k_from_z(self) -> int:
        # cost of initializing best_sample is zero
        z_loop = self.rand_args.n_osc * 3 + 1 # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        return int(self.max_z_ops // z_loop)

    def init_best_sample(self) -> sample.Sample:
        return self.gen_empty_sample()

    def draw_temp_sample(self, *args, **kwargs) -> sample.Sample:
        return self.draw_sample()

class MCOneShotWeight(MCOneShot):
    """Use the MCOneShot algorithm but only draw new weights for each sample"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 3 + 1   # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        z_loop = self.rand_args.n_osc + 1       # draw a sample with n new weights and one offset each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        return self.draw_sample_weights(base_sample)



class MCExploit(MonteCarlo):
    """This algorithm most closely resembles a strictly guided random-walk with random step size.
    The walk is in n-dimensional space where n is the number of oscillators.
    A step is only taken if the loss decreases, hence strictly guided.
    Heuristic is not an appropriate term as selection (which weight or amount) is random, not guided by a heuristic.
    The step size varies on each step and is drawn from a normal or uniform distribution.
    We could also say that MCExploit is simulated annealing without temperature and deterministic acceptance.
    The algorithm remembers the best solution to date.
    The to-date best solution is returned when allocated resources (z-ops) are exhausted."""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 3 + 1 # draw a new sample with n weighted oscillators (phase + frequency + gain (weight) --> 3) on each loop and an offset (bias) --> 1
        z_loop = self.j_replace * 3 # j weights (gain) and oscillators (phase+frequency) or the offset updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_sample(base_sample, osc_to_replace)

class MCExploitWeight(MCExploit):
    """Use the MCExploit algorithm but only draw new weights for each sample"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 3 + 1 # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        z_loop = self.j_replace # j weights are updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_sample_weights(base_sample, osc_to_replace)

# TODO: during the initialization phase, the position of replaced oscillators may better be random,
# only must memory be kept of which has been updated, and the list from which to draw must be updated to shrink
class MCExploitFast(MCExploit):
    """algorithm is equivalent to MCExploit after an initialization phase.
    the duration of the initialization phase is non deterministic and inspired by las vegas algorithms.
    the algorithm combines fast convergence speed of las vegas algorithms with the exploitation of MCExploit.
    the MCExploitFast algorithm continues to be deterministic in runtime.
    
    loop over oscillators from top to bottom,
    once an oscillator has been replaced on each row,
    proceed to replace oscillators in random positions
    """

    def __init__(self, algo_args: algarty.AlgoArgs):
        super().__init__(algo_args)
        self.changed_once: List[bool] = [False for _ in range(self.rand_args.n_osc + 1)] # + 1 for offset

    def get_osc_to_replace(self) -> List[int]:
        if all(self.changed_once):
            return self.draw_random_indices(self.j_replace)
        else:
            # find first False and get index
            first_false = next(i for (i, val) in enumerate(self.changed_once) if val == False)
            # wrap around max index
            osc_to_replace = [i % len(self.changed_once) for i in range(first_false, first_false+self.j_replace)]
            return np.array(osc_to_replace)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        osc_to_replace = self.get_osc_to_replace()
        return self.draw_partial_sample(base_sample, osc_to_replace)

    def comp_samples(self, base_sample: sample.Sample, temp_sample: sample.Sample) -> sample.Sample:
        if temp_sample.rmse < base_sample.rmse:
            if not all(self.changed_once):
                # set changed_once to True for all oscillators that have been replaced
                first_false = next(i for (i, val) in enumerate(self.changed_once) if val == False)
                # can't exceed max index
                true_len = min(self.j_replace, len(self.changed_once) - first_false)
                self.changed_once[first_false] = [True] * true_len
            return temp_sample
        return base_sample

        

class MCAnneal(MonteCarlo):
    """use a schedule to reduce the number of oscillators across iterations.
    akin to simulated annealing"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 3 + 1 # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        z_loop = self.rand_args.n_osc * 1.5 + 1 # n oscillators (phase+frequency+gain) and the offset updated on first loop
        # however we have a linear schedule with the replaced number of oscillator decreasing over time
        # therefore on average we count 3 * n_osc / 2 + 1 -> 1.5 * n_osc +1
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
        z_init = self.rand_args.n_osc * 3 + 1 # initialize a best sample with n oscillators (freq [n*1] and phase [n*1]), n weights [n*1] and an offset (bias) [1]
        z_loop = self.rand_args.n_osc * 0.5 # we only draw new weights and omit * 2 for oscillators
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, k: int, *args, **kwargs) -> sample.Sample:
        j_replace = self.read_j_from_schedule(k)
        osc_to_replace = self.draw_random_indices(j_replace)
        return self.draw_partial_sample_weights(base_sample, osc_to_replace)

class MCAnnealLog(MCAnneal):
    """use logarithmic schedule to reduce the number of oscillators across iterations."""

    def read_j_from_schedule(self, k: int) -> int:
        """logarithmic schedule"""
        temperature = (1 - k / self.k_samples)**np.e
        return np.ceil(self.rand_args.n_osc*temperature).astype(int)

class MCAnnealLogWeight(MCAnnealWeight):
    """use logarithmic schedule and optimize only weights"""

    def read_j_from_schedule(self, k: int) -> int:
        """logarithmic schedule"""
        temperature = (1 - k / self.k_samples)**np.e
        return np.ceil(self.rand_args.n_osc*temperature).astype(int)



class MCGrowShrink(MonteCarlo):
    """generalization over MCDampen
    where we have l-the probability to dampen vs 1-l the probability to grow
    and then we have j, the number of weights to adapt
    and lastly we have h the factor by which to dampen or grow weights"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 3 + 1 # initialize a best sample with n oscillators, n weights and an offset (bias)
        z_loop = self.j_replace # j weights (or bias) are updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        temp_sample = copy.deepcopy(base_sample)
        rng = np.random.default_rng() if self.mp else const.RNG # each rng needs to be seeded differently for multiprocessing
        # for example, if l_dampen = 0.9 then we dampen 90% of the time
        if self.l_damp_prob > rng.uniform():
            multiplier = self.h_damp_fac
        else:
            multiplier = 1 / self.h_damp_fac # growth factor is inverse of dampening factor
        osc_to_replace, change_offset = self.draw_weight_indices_or_offset(self.j_replace)
        
        if change_offset:
            temp_sample.offset *= multiplier
        temp_sample.weights[osc_to_replace] *= multiplier
        return temp_sample

class MCDampen(MCGrowShrink):
    """generalization over MCPurge
    initialize pool of n oscillators and weights
    take j weights, multiply weights by dampening-factor h, and evaluate the sample
    where h >= 0 and <= 1
    accept change if rmse is lower
    stop when z_ops is exhausted

    MCPurge is a special case of MCDampen with h=0
    """

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        temp_sample = copy.deepcopy(base_sample)
        osc_to_replace, change_offset = self.draw_weight_indices_or_offset(self.j_replace)
        if change_offset:
            temp_sample.offset *= self.h_damp_fac
        temp_sample.weights[osc_to_replace] *= self.h_damp_fac
        return temp_sample

class MCPurge(MCDampen):
    """
    initialize pool of n oscillators and weights
    set j weights to 0 and evaluate the sample
    accept change if rmse is lower
    stop after looping over each oscillator or when z_ops is exhausted
    # TODO: stop after looping over each oscillator
    """

    def __init__(self, algo_args: algarty.AlgoArgs):
        super().__init__(algo_args)
        assert self.h_damp_fac == 0, "h_damp_fac must be 0 for MCPurge"


# TODO: proper implementation of Simulated annealing with probability of acceptance of worse solutions
# TODO: implement proper Metropolis-Hastings algorithm


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
        
    def search(self, *args, **kwargs):
        """
        
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
            weighted_sum = np.sum(best_sample.signal_matrix.T * best_sample.weights, axis=1) + offset
            return data_analysis.compute_rmse(weighted_sum, self.target)

        def eval_func_weight(weights):
            """adapt weights for simulated annealing"""
            weighted_sum = np.sum(best_sample.signal_matrix.T * weights, axis=1) + best_sample.offset
            return data_analysis.compute_rmse(weighted_sum, self.target)

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

        # optimize the offset before optimizing the weights since SciPy's Bounds class is too restrictive to optimize both simultaneously
        # the documentation may also just be too confusing at explaining how to do this
        lo, hi = self.rand_args.offset_dist.get_low_high()
        offset_bounds = Bounds(lo, hi)
        niter = int(self.max_z_ops // self.rand_args.n_osc) // 20
        result = basinhopping(eval_func_offset, best_sample.offset, minimizer_kwargs={"method": "COBYLA"}, niter=niter, accept_test=offset_bounds, seed=const.GLOBAL_SEED)
        best_sample.offset = result.x

        # optimize weights
        lo, hi = self.rand_args.weight_dist.get_low_high()
        weight_bounds = Bounds(lo, hi)
        # using a factor of 1/10, the runtime of the algorithm is similar to that of other algorithms
        niter = int(self.max_z_ops // self.rand_args.n_osc) // 10
        result = basinhopping(eval_func_weight, best_sample.weights, minimizer_kwargs={"method": "COBYLA"}, niter=niter, accept_test=weight_bounds, seed=const.GLOBAL_SEED)
        best_sample.weights = result.x

        best_sample.update(self.target)
        self.z_ops += niter * self.rand_args.n_osc
        
        return best_sample, self.z_ops

