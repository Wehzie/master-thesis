import copy
from pathlib import Path
import gen_signal_python
import algo
import sample
import param_types as party
import data_analysis
import const
rng = const.RNG

from typing import Tuple

from tqdm import tqdm

class MCOneShot(algo.SearchAlgo):
    """monte carlo algorithm for samples consisting of independent oscillators"""

    def init_best_sample(self) -> sample.Sample:
        return self.gen_empty_sample()

    def draw_temp_sample(self, *args, **kwargs) -> sample.Sample:
        return self.draw_sample()

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
            temp_sample = self.draw_temp_sample(best_sample)
            best_sample = self.comp_samples(best_sample, temp_sample)
            self.manage_state(temp_sample, k)

        return best_sample, self.z_ops

class MCOneShotWeight(MCOneShot):
    """Use the MCOneShot algorithm but only draw new weights for each sample"""

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample, *args, **kwargs) -> sample.Sample:
        return self.draw_sample_weights(base_sample)

class MCExploit(algo.SearchAlgo):
    """monte carlo algorithm exploiting a single sample by iterative re-draws
    
    self.mp must be set to True or False"""

    def init_best_sample(self) -> sample.Sample:
        return self.draw_sample()

    def draw_temp_sample(self, base_sample: sample.Sample) -> sample.Sample:
        return self.draw_partial_sample(base_sample)

    def search(self, *args, **kwargs):
        self.clear_state()
        self.handle_mp(kwargs)

        best_sample = self.init_best_sample()
        for k in tqdm(range(self.k_samples)):
            temp_sample = self.draw_temp_sample(best_sample)
            best_sample = self.comp_samples(best_sample, temp_sample)
            self.manage_state(temp_sample, k)

        return best_sample, self.z_ops
