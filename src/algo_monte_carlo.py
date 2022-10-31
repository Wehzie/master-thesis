from abc import abstractclassmethod
from typing import Tuple

import algo
import sample

import numpy as np
from tqdm import tqdm

# TODO: check that infer k from z_ops is correct for each algorithm

class MonteCarlo(algo.SearchAlgo):
    """abstract class"""

    def search(self, *args, **kwargs) -> Tuple[sample.Sample, int]:
        """generate k-signals which are a sum of n-oscillators
        on each iteration draw a new full model (matrix of n-oscillators)
        
        params:
            k_samples: number of times to draw a matrix
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model
            history: whether to store all generated samples, may run out of RAM
            args_path: when not none, write history to disk instead of RAM at specified path
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

    def init_best_sample(self) -> sample.Sample:
        return self.gen_empty_sample()

    def draw_temp_sample(self, *args, **kwargs) -> sample.Sample:
        return self.draw_sample()

class MCOneShotWeight(MCOneShot):
    """Use the MCOneShot algorithm but only draw new weights for each sample"""

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        return self.draw_sample_weights(base_sample)

class MCExploit(MonteCarlo):
    """monte carlo algorithm exploiting a single sample by iterative re-draws
    
    self.mp must be set to True or False"""

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        return self.draw_partial_sample(base_sample, self.j_replace)

class MCExploitWeight(MCExploit):
    """Use the MCExploit algorithm but only draw new weights for each sample"""

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        return self.draw_partial_sample_weights(base_sample, self.j_replace)

class MCAnneal(MonteCarlo):
    """use a schedule to reduce the number of oscillators across iterations.
    akin to simulated annealing"""
    
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
        return self.draw_partial_sample(base_sample, j_replace)

class MCAnnealWeight(MCAnneal):
    """use the MCAnneal algorithm but only draw new weights for each sample"""

    def draw_temp_sample(self, base_sample: sample.Sample, k: int, *args, **kwargs) -> sample.Sample:
        j_replace = self.read_j_from_schedule(k)
        return self.draw_partial_sample_weights(base_sample, j_replace)