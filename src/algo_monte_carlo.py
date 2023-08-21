"""
This module implements Monte Carlo algorithms for optimization.

Monte Carlo algorithms have deterministic runtime and draw from randomness to find the best solution.

Monte Carlo algorithms form a class that is relatively easy to implement in hardware.
No gradient is propagated nor must complex memory be maintained.
For this reason, Monte Carlo algorithms are a good starting point for neuromorphic hardware.
"""

import copy
from typing import Tuple, List

import algo
import sample
import const

import algo_args_type as algarty
import gen_signal

import numpy as np
from tqdm import tqdm


def debug(sample: sample.Sample):
    """print debug information about a sample"""
    m = np.mean(sample.weighted_sum)
    m2 = np.mean(np.sum(sample.signal_matrix, axis=0))
    mean10rows = [np.mean(row) for row in sample.signal_matrix[:10, :]]
    o = sample.offset
    maximum_matrix = np.max(sample.signal_matrix)
    minimum_matrix = np.min(sample.signal_matrix)
    maximum_weighted_sum = np.max(sample.weighted_sum)
    minimum_weighted_sum = np.min(sample.weighted_sum)

    print("\n\n")
    print(f"shape_matrix: {sample.signal_matrix.shape}")
    print(f"shape_weighted_sum: {sample.weighted_sum.shape}")
    print(f"mean_weighted_sum: {m}, mean_matrix: {m2}, offset: {o}")
    print(f"mean10rows: {mean10rows}")
    print(f"max_matrix: {maximum_matrix}, min_matrix: {minimum_matrix}")
    print(f"max_weighted_sum: {maximum_weighted_sum}, min_weighted_sum: {minimum_weighted_sum}")
    exit()


class MonteCarlo(algo.SearchAlgo):
    """abstract class for Monte Carlo algorithms"""

    def search(self, *args, **kwargs) -> Tuple[sample.Sample, int]:
        """
        generate k-signals which are a sum of n-oscillators

        on each iteration draw a new full model (matrix of n-oscillators)
        """
        self.clear_state()
        self.handle_mp(kwargs)

        best_sample = self.init_best_sample()
        for k in tqdm(range(self.k_samples)):
            temp_sample = self.draw_temp_sample(best_sample, k)
            best_sample = self.comp_samples(best_sample, temp_sample, k)
            self.manage_state(temp_sample, k)

        return best_sample, self.z_ops


class MCOneShot(MonteCarlo):
    """
    This algorithms most closely resembles brute-force search.

    Compared to brute-force search, this algorithm remembers it's best solution to date.
    The to-date best solution is returned when allocated resources (z-ops) are exhausted.
    """

    def infer_k_from_z(self) -> int:
        """infer the number of iterations (k) from the maximum number of perturbations (z)"""
        # cost of initializing best_sample is zero
        z_loop = (
            self.rand_args.n_osc * 3 + 1
        )  # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        return int(self.max_z_ops // z_loop)

    def init_best_sample(self) -> sample.Sample:
        """draw an empty oscillator ensemble"""
        return self.gen_empty_sample()

    def draw_temp_sample(self, *args, **kwargs) -> sample.Sample:
        """draw a random oscillator ensemble"""
        return self.draw_sample()


class MCOneShotWeight(MCOneShot):
    """Use the MCOneShot algorithm but only draw new weights for each sample"""

    def infer_k_from_z(self) -> int:
        """infer the number of iterations (k) from the maximum number of perturbations (z)"""
        z_init = (
            self.rand_args.n_osc * 3 + 1
        )  # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        z_loop = (
            self.rand_args.n_osc + 1
        )  # draw a sample with n new weights and one offset each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        """draw a random oscillator ensemble"""
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """perturb the base sample and return the perturbed sample"""
        return self.draw_sample_weights(base_sample)


class MCExploit(MonteCarlo):
    """
    This algorithm most closely resembles a strictly guided random-walk with random step size.

    The walk is in n-dimensional space where n is the number of oscillators.
    A step is only taken if the loss decreases, hence strictly guided.
    Heuristic is not an appropriate term as selection (which weight or amount) is random, not guided by a heuristic.
    The step size varies on each step and is drawn from a normal or uniform distribution.
    We could also say that MCExploit is simulated annealing without temperature and deterministic acceptance.
    The algorithm remembers the best solution to date.
    The to-date best solution is returned when allocated resources (z-ops) are exhausted.
    """

    def infer_k_from_z(self) -> int:
        """infer the number of iterations (k) from the maximum number of perturbations (z)"""
        z_init = (
            self.rand_args.n_osc * 3 + 1
        )  # draw a new sample with n weighted oscillators (phase + frequency + gain (weight) --> 3) on each loop and an offset (bias) --> 1
        z_loop = (
            self.j_replace * 3
        )  # j weights (gain) and oscillators (phase+frequency) or the offset updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        """draw a random oscillator ensemble"""
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """perturb the base sample and return the perturbed sample"""
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_sample(base_sample, osc_to_replace)


class MCExploitJ10(MCExploit):
    """Use the MCExploit algorithm but replace 10 oscillators per sample"""

    pass


class MCExploitDecoupled(MCExploit):
    """
    decouple weight and oscillator shocks.

    pick oscillator, pick whether to change weight or oscillator, change, accept or reject, repeat.
    """

    def infer_k_from_z(self) -> int:
        """compute the number of iterations (k) that can be performed with the allocated resources (z-ops)"""
        z_init = (
            self.rand_args.n_osc * 3 + 1
        )  # draw a new sample with n weighted oscillators (phase + frequency + gain (weight) --> 3) on each loop and an offset (bias) --> 1
        z_loop = (
            self.j_replace * 1.5
        )  # j weights (gain) [1] and oscillators (phase+frequency [2]) or the offset [1] updated on each loop
        # on average that's 1.5 operations per loop
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """perturb the base oscillator ensemble and return the perturbed ensemble"""
        osc_to_replace = self.draw_random_indices(self.j_replace)
        temp_sample = self.draw_partial_sample(base_sample, osc_to_replace)

        # offset is passed as number_of_oscillators+1, can be removed once the sample has been drawn
        osc_to_replace, _ = gen_signal.SignalGenerator.separate_oscillators_from_offset(
            osc_to_replace, self.rand_args.n_osc
        )

        # throw coin to decide whether to replace oscillators or weights and offset
        replace_oscillator = const.RNG.uniform() < 0.5
        replace_weight_offset = not replace_oscillator

        if replace_oscillator:  # set weights and offset back to their original state
            temp_sample.weights[osc_to_replace] = base_sample.weights[osc_to_replace]
            temp_sample.offset = base_sample.offset
        elif replace_weight_offset:  # set the oscillator back to its original state
            temp_sample.signal_matrix[osc_to_replace,] = base_sample.signal_matrix[osc_to_replace,]
        return temp_sample


class MCExploitWeight(MCExploit):
    """Use the MCExploit algorithm but only draw new weights for each sample"""

    def infer_k_from_z(self) -> int:
        """compute the number of iterations k from the number of operations z"""
        z_init = (
            self.rand_args.n_osc * 3 + 1
        )  # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        z_loop = self.j_replace  # j weights are updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """perturb the base sample and return the perturbed sample"""
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_sample_weights(base_sample, osc_to_replace)


class MCExploitNeighborWeight(MCExploitWeight):
    """Draws candidate weights from a Gaussian around a weight and its neighbors"""

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """draw a temporary sample, a neighbor of the base sample"""
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_weight_neighbor(base_sample, osc_to_replace)


class MCExploitFast(MCExploit):
    """
    algorithm is equivalent to MCExploit after an initialization phase.

    the duration of the initialization phase is non deterministic and inspired by las vegas algorithms.
    the algorithm combines fast convergence speed of las vegas algorithms with the exploitation of MCExploit.
    the MCExploitFast algorithm continues to be deterministic in runtime.

    loop over oscillators from top to bottom,
    once an oscillator has been replaced on each row,
    proceed to replace oscillators in random positions
    """

    def __init__(self, algo_args: algarty.AlgoArgs):
        if algo_args is None:
            return
        super().__init__(algo_args)
        self.changed_once: List[bool] = [
            False for _ in range(self.rand_args.n_osc + 1)
        ]  # + 1 for offset

    def get_osc_to_replace(self) -> List[int]:
        """select which oscillators should be perturbed"""
        if all(self.changed_once):
            return self.draw_random_indices(self.j_replace)
        else:
            # find first False and get index
            first_false = next(i for (i, val) in enumerate(self.changed_once) if val == False)
            # wrap around max index
            osc_to_replace = [
                i % len(self.changed_once) for i in range(first_false, first_false + self.j_replace)
            ]
            return np.array(osc_to_replace)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """perturb the base sample and return the perturbed sample"""
        osc_to_replace = self.get_osc_to_replace()
        return self.draw_partial_sample(base_sample, osc_to_replace)

    def comp_samples(
        self, base_sample: sample.Sample, temp_sample: sample.Sample, *args, **kwargs
    ) -> sample.Sample:
        """compare base and temp sample and return the better one"""
        if temp_sample.rmse < base_sample.rmse:
            if not all(self.changed_once):
                # set changed_once to True for all oscillators that have been replaced
                first_false = next(i for (i, val) in enumerate(self.changed_once) if val == False)
                # can't exceed max index
                true_len = min(self.j_replace, len(self.changed_once) - first_false)
                self.changed_once[first_false] = [True] * true_len
            return temp_sample
        return base_sample


class MCOscillatorAnneal(MonteCarlo):
    """
    use a schedule to reduce the number of oscillators across iterations.

    akin to simulated annealing
    """

    def infer_k_from_z(self) -> int:
        """return the number of iterations to run the algorithm for"""
        z_init = (
            self.rand_args.n_osc * 3 + 1
        )  # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        z_loop = (
            self.rand_args.n_osc * 1.5 + 1
        )  # n oscillators (phase+frequency+gain) and the offset updated on first loop
        # however we have a linear schedule with the replaced number of oscillator decreasing over time
        # therefore on average we count 3 * n_osc / 2 + 1 -> 1.5 * n_osc +1
        return int((self.max_z_ops - z_init) // z_loop)

    def read_j_from_schedule(self, k: int) -> int:
        """
        return the number of oscillator or weights to draw according to a linear schedule

        params:
            k: iteration number

        returns:
            j: number of oscillators or weights to draw
        """
        temperature = 1 - k / self.k_samples
        return np.ceil(self.rand_args.n_osc * temperature).astype(int)

    def init_best_sample(self) -> sample.Sample:
        """draw a random oscillator ensemble"""
        return self.draw_sample()

    def draw_temp_sample(
        self, base_sample: sample.Sample, k: int, *args, **kwargs
    ) -> sample.Sample:
        """return a perturbed version of the base ensemble"""
        j_replace = self.read_j_from_schedule(k)
        osc_to_replace = self.draw_random_indices(j_replace)
        return self.draw_partial_sample(base_sample, osc_to_replace)


class MCOscillatorAnnealWeight(MCOscillatorAnneal):
    """use the MCAnneal algorithm but only draw new weights for each sample"""

    def infer_k_from_z(self) -> int:
        """return the number of iterations to run the algorithm for"""
        z_init = (
            self.rand_args.n_osc * 3 + 1
        )  # initialize a best sample with n oscillators (freq [n*1] and phase [n*1]), n weights [n*1] and an offset (bias) [1]
        z_loop = self.rand_args.n_osc * 0.5  # we only draw new weights and omit * 2 for oscillators
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(
        self, base_sample: sample.Sample, k: int, *args, **kwargs
    ) -> sample.Sample:
        """perturb the base oscillator ensemble"""
        j_replace = self.read_j_from_schedule(k)
        osc_to_replace = self.draw_random_indices(j_replace)
        return self.draw_partial_sample_weights(base_sample, osc_to_replace)


class MCOscillatorAnnealLog(MCOscillatorAnneal):
    """use logarithmic schedule to reduce the number of oscillators across iterations."""

    def read_j_from_schedule(self, k: int) -> int:
        """logarithmic schedule"""
        temperature = (1 - k / self.k_samples) ** np.e
        return np.ceil(self.rand_args.n_osc * temperature).astype(int)


class MCOscillatorAnnealLogWeight(MCOscillatorAnnealWeight):
    """use logarithmic schedule and optimize only weights"""

    def read_j_from_schedule(self, k: int) -> int:
        """logarithmic schedule"""
        temperature = (1 - k / self.k_samples) ** np.e
        return np.ceil(self.rand_args.n_osc * temperature).astype(int)


class MCGrowShrink(MonteCarlo):
    """
    generalization over MCDampen

    where we have l-the probability to dampen vs 1-l the probability to grow
    and then we have j, the number of weights to adapt
    and lastly we have h the factor by which to dampen or grow weights
    """

    def infer_k_from_z(self) -> int:
        """compute the number of iterations to run the algorithm for"""
        z_init = (
            self.rand_args.n_osc * 3 + 1
        )  # initialize a best sample with n oscillators, n weights and an offset (bias)
        z_loop = self.j_replace  # j weights (or bias) are updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def init_best_sample(self) -> sample.Sample:
        """draw a random oscillator ensemble"""
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """perturb the base ensemble by growing or shrinking j weights with factor h"""
        temp_sample = copy.deepcopy(base_sample)
        rng = (
            np.random.default_rng() if self.mp else const.RNG
        )  # each rng needs to be seeded differently for multiprocessing
        # for example, if l_dampen = 0.9 then we dampen 90% of the time
        if self.l_damp_prob > rng.uniform():
            multiplier = self.h_damp_fac
        else:
            multiplier = 1 / self.h_damp_fac  # growth factor is inverse of dampening factor
        osc_to_replace, change_offset = self.draw_weight_indices_or_offset(self.j_replace)

        if change_offset:
            temp_sample.offset *= multiplier
        temp_sample.weights[osc_to_replace] *= multiplier
        return temp_sample


class MCDampen(MCGrowShrink):
    """
    generalization over MCPurge

    initialize pool of n oscillators and weights
    take j weights, multiply weights by dampening-factor h, and evaluate the sample
    where h >= 0 and <= 1
    accept change if rmse is lower
    stop when z_ops is exhausted

    MCPurge is a special case of MCDampen with h=0
    """

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """perturb the base ensemble by setting j weights to 0"""
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
        if algo_args is None:
            return
        super().__init__(algo_args)
        assert self.h_damp_fac == 0, "h_damp_fac must be 0 for MCPurge"
