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
import param_types as party

import numpy as np
from tqdm import tqdm
from scipy.optimize import basinhopping

# TODO: ensure that offset is treated in the same way as weights

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

    # TODO: evaluate whether redrawing an oscillator is equally expensive to drawing a weight
    # right now drawing a weight costs z=1
    # the oscillator algos consistently outperform
    # maybe this is only because essentially it's like drawing a weight, frequency, offset, and phase in one
    # so we'd have to consider z=4
    # another thought is to consider number of evaluations as z, this is used by scipy
    # lastly, maybe this issue isn't so important, since we care about comparing the algorithms
    # and not an algorithms weight only vs weight and oscillator version
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

    def __init__(self, algo_args: party.AlgoArgs):
        super().__init__(algo_args)
        self.changed_once: List[bool] = [False for _ in range(self.rand_args.n_osc)]

    def get_osc_to_replace(self) -> List[int]:
        if all(self.changed_once):
            return self.draw_random_indices(self.j_replace)
        else:
            # find first False and get index
            first_false = next(i for (i, val) in enumerate(self.changed_once) if val == False)
            # wrap around max index
            osc_to_replace = [i % self.rand_args.n_osc for i in range(first_false, first_false+self.j_replace)]
            return osc_to_replace

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        osc_to_replace = self.get_osc_to_replace()
        return self.draw_partial_sample(base_sample, osc_to_replace)

    def comp_samples(self, base_sample: sample.Sample, temp_sample: sample.Sample) -> sample.Sample:
        if temp_sample.rmse < base_sample.rmse:
            if not all(self.changed_once):
                # set changed_once to True for all oscillators that have been replaced
                first_false = next(i for (i, val) in enumerate(self.changed_once) if val == False)
                # can't exceed max index
                true_len = min(self.j_replace, self.rand_args.n_osc - first_false)
                self.changed_once[first_false] = [True] * true_len
            return temp_sample
        return base_sample

        

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
        z_init = self.rand_args.n_osc * 2
        z_loop = self.j_replace # j weights are updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        rng = np.random.default_rng() if self.mp else const.RNG # each rng needs to be seeded differently for multiprocessing
        # l_dampen = 0.9, then 90% of the time we dampen
        if self.l_damp_prob > rng.uniform():
            multiplier = self.h_damp_fac
        else:
            multiplier = 1 / self.h_damp_fac # growth factor is inverse of dampening factor
        osc_to_replace = self.draw_random_indices(self.j_replace)
        temp_sample = copy.deepcopy(base_sample)
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
        osc_to_replace = self.draw_random_indices(self.j_replace)
        temp_sample = copy.deepcopy(base_sample)
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

    def __init__(self, algo_args: party.AlgoArgs):
        super().__init__(algo_args)
        assert self.h_damp_fac == 0, "h_damp_fac must be 0 for MCPurge"



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

        def eval_func(weights):
            """adapt weights for simulated annealing"""
            weighted_sum = np.sum(best_sample.signal_matrix.T * weights, axis=1)
            return data_analysis.compute_rmse(weighted_sum, self.target)

        best_sample = self.init_best_sample()
        
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

        lo, hi = self.rand_args.weight_dist.get_low_high()
        weight_bounds = Bounds(lo, hi)

        # using a factor of 1/10, the runtime of the algorithm is similar to that of other algorithms
        niter = int(self.max_z_ops // self.rand_args.n_osc) // 10

        result = basinhopping(eval_func, best_sample.weights, minimizer_kwargs={"method": "COBYLA"}, niter=niter, accept_test=weight_bounds, seed=const.GLOBAL_SEED)

        best_sample.weights = result.x
        best_sample.update(self.target)
        self.z_ops += niter * self.rand_args.n_osc
        
        return best_sample, self.z_ops

