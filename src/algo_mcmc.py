import algo_monte_carlo
import sample
import const

import numpy as np

class MCExploitErgodic(algo_monte_carlo.MCExploit):
    """An ergodic version of the MCExploit algorithm.
    
    Compared to MCExploit, this algorithm accepts worse samples with a small probability"""

    def accept_candidate_sample(self, rmse_base: float, rmse_temp: float, temperature: float = 0.1) -> bool:
        """Accept a new sample when according to some rules.

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

class MCAnneal(algo_monte_carlo.MonteCarlo):

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 3 + 1 # draw a new sample with n weighted oscillators (phase + frequency + gain (weight) --> 3) on each loop and an offset (bias) --> 1
        z_loop = self.j_replace * 3 # j weights (gain) and oscillators (phase+frequency) or the offset updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """draw a temporary sample, a neighbor of the base sample"""
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_sample(base_sample, osc_to_replace)
    
    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def get_temp(self, k: int) -> float:
        """compute temperature for k-th iteration"""
        return 1 - (k+1) / self.k_samples

    def accept_candidate_sample(self, base_rmse: float, temporary_rmse: float, temperature: float) -> bool:
        """define the acceptance function follow the definition of Kirkpatrick et al. (1983)"""
        if temporary_rmse < base_rmse:
            return True
        acceptance_ratio = np.exp(-(temporary_rmse - base_rmse) / temperature)
        return const.RNG.uniform(0, 1) <= acceptance_ratio

    def comp_samples(self, base_sample: sample.Sample, temp_sample: sample.Sample, iteration_k: int) -> sample.Sample:
        temperature = self.get_temp(iteration_k)
        if self.accept_candidate_sample(base_sample.rmse, temp_sample.rmse, temperature):
            return temp_sample
        return base_sample

class MCAnnealWeight(MCAnneal):
    """Regular simulated annealing following Kirkpatrick et al. (1983)"""

    def infer_k_from_z(self) -> int:
        z_init = self.rand_args.n_osc * 3 + 1 # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        z_loop = self.j_replace # j weights are updated on each loop
        return int((self.max_z_ops - z_init) // z_loop)

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        """draw a temporary sample, a neighbor of the base sample"""
        osc_to_replace = self.draw_random_indices(self.j_replace)
        return self.draw_partial_weight_neighbor_sample(base_sample, osc_to_replace)
    
