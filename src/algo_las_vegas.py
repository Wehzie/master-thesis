import copy
from pathlib import Path
import gen_signal_python
import data_analysis
import param_types as party
import algo
import sample
import const

from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

class LasVegas(algo.SearchAlgo):

    def init_las_vegas(self, weight_init: Union[None, str], store_det_args: bool) -> sample.Sample:
        if weight_init is None: # in this mode, weights are not adapted
            signal_matrix = np.zeros((self.rand_args.n_osc, self.rand_args.samples))
            signal_args = list()
        else:
            signal_matrix, signal_args = gen_signal_python.sum_atomic_signals(self.rand_args, store_det_args)

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
            weights = None # in this mode, weights are not adapted

        return sample.Sample(signal_matrix, weights, np.sum(signal_matrix, axis=0), 0, np.inf, signal_args)
    
    def infer_k_from_z(self) -> int:
        return None

    def init_best_sample(self) -> sample.Sample:
        return super().init_best_sample()

    def draw_temp_sample(self) -> sample.Sample:
        return super().draw_temp_sample()

    def comp_samples(self, base_sample: sample.Sample, temp_sample: sample.Sample) -> Tuple[sample.Sample, bool]:
        """compare two samples and return the one with lower rmse
        
        returns:
            sample: the sample with lower rmse
            changed: true if the new sample is better than the old one, false if the old one is better"""
        if temp_sample.rmse < base_sample.rmse:
            return temp_sample, True
        return base_sample, False

    def search(self, *args, **kwargs) -> Tuple[sample.Sample, int]:
        """generate k-signals which are a sum of n-oscillators
        model is constructed by aggregation
        oscillator candidates are accepted into the model when they lower RMSE

        returns:
            best_sample: the best sample found
            z_ops: the number of operations performed
        """
        self.clear_state()
        self.handle_mp(kwargs)

        def generator(): # support tqdm timer and iterations per second
            while not self.stop_on_z_ops(): yield

        best_sample = self.gen_zero_sample()
        for k in tqdm(generator()):
            base_sample = self.gen_zero_sample()
            i = 0 # number of replaced weights
            while i < self.rand_args.n_osc and not self.stop_on_z_ops():
                temp_sample = self.draw_partial_sample(base_sample, [i])
                base_sample, changed = self.comp_samples(base_sample, temp_sample)
                if changed: i += 1 # move to next row
                self.z_ops += 2
            
            self.manage_state(base_sample, k)
            best_sample, _ = self.comp_samples(best_sample, base_sample)

        return best_sample, self.z_ops

class LasVegasWeight(algo.SearchAlgo):
    def draw_weight(self, weights: np.ndarray, i: int):
        w = self.rand_args.weight_dist.draw()
        weights[i] = w
        return weights

    def draw_las_vegas_weight_candidate(self, base_sample: sample.Sample, i):
        temp_sample = copy.deepcopy(base_sample)
        temp_sample.weights = self.draw_weight(temp_sample.weights, i)
        temp_sample.signal_sum = sample.Sample.predict(temp_sample.signal_matrix, temp_sample.weights, 0)
        temp_sample.rmse = data_analysis.compute_rmse(temp_sample.signal_sum, self.target)
        return temp_sample

    def search(self,
        k_samples: int,
        weight_init: Union[None, str],
        store_det_args: bool = False,
        history: bool = False,
        args_path: Path = None) -> Tuple[sample.Sample, int]:

        best_sample = self.init_las_vegas(weight_init, store_det_args)
        best_sample_j = None
        z_ops = 0
        for _ in tqdm(range(k_samples)):
            base_sample = self.init_las_vegas(weight_init, store_det_args)
            i, j = 0, 0 # number of accepted and drawn weights respectively
            while i < self.rand_args.n_osc:
                temp_sample = self.draw_las_vegas_weight_candidate(base_sample, i)
                base_sample, i, _ = self.eval_las_vegas(base_sample, temp_sample, i, None, None)
                j += 1

            z_ops += j
            best_sample, _, best_sample_j = self.eval_las_vegas(base_sample, best_sample, 0, j, best_sample_j)
            
        best_sample.signal_matrix = (best_sample.signal_matrix.T * best_sample.weights).T # TODO: change sample type
        self.samples.append(best_sample)
        return best_sample, z_ops