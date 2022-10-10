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

    def draw_las_vegas_candidate(self, base_sample: sample.Sample, i: int, mod_args: party.PythonSignalRandArgs, store_det_args: bool, history: bool):
        # TODO: this can be done more efficiently without deepcopy and recomputing all
        temp_sample = copy.deepcopy(base_sample)
        signal, signal_args = gen_signal_python.sum_atomic_signals(mod_args, store_det_args)
        signal = signal.flatten()
        temp_sample.signal_matrix[i,:] = signal
        if store_det_args or history:
            signal_args = signal_args[0]
            temp_sample.signal_args.append(signal_args)
        temp_sample.signal_sum = np.sum(temp_sample.signal_matrix, axis=0)
        temp_sample.rmse = data_analysis.compute_rmse(temp_sample.signal_sum, self.target)
        return temp_sample

    def eval_las_vegas(self, temp_sample: sample.Sample, base_sample: sample.Sample, i: int, j: int, best_sample_j: int
    ) -> Tuple[sample.Sample, int, int]:
        if temp_sample.rmse < base_sample.rmse:
            base_sample = temp_sample
            return temp_sample, i+1, j
        return base_sample, i, best_sample_j

    def search(self,
        k_samples: int,
        weight_init: Union[None, str],
        store_det_args: bool = False,
        history: bool = False,
        args_path: Path = None) -> Tuple[sample.Sample, int]:
        """generate k-signals which are a sum of n-oscillators
        model is constructed by aggregation
        oscillator candidates are accepted into the model when they lower RMSE

        param:
            k_samples: number of times to draw a matrix
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model
            history: whether to store all generated samples, may run out of RAM
            args_path: when not none, write history to disk instead of RAM at specified path

        return:
            best_sample, best_model_j
        """
        if history and args_path: args_path.unlink(missing_ok=True) # delete file if it exists

        # TODO: hacky solution to draw 1 oscillator from sum_atomic signals
        mod_args = copy.deepcopy(self.rand_args)
        mod_args.n_osc = 1

        best_sample = self.init_las_vegas(weight_init, store_det_args)
        best_sample_j = None
        z_ops = 0
        for ki in tqdm(range(k_samples)):
            base_sample = self.init_las_vegas(weight_init, store_det_args) # build up a signal_matrix in here
            i, j = 0, 0 # number of accepted and drawn weights respectively
            while i < self.rand_args.n_osc: # construct a model
                temp_sample = self.draw_las_vegas_candidate(base_sample, i, mod_args, store_det_args, history)
                base_sample, i, _ = self.eval_las_vegas(temp_sample, base_sample, i, None, None)
                j += 1
            
            z_ops += j

            self.manage_state(base_sample, k_samples, ki, history, args_path)
            best_sample, _, best_sample_j = self.eval_las_vegas(base_sample, best_sample, 0, j, best_sample_j)
            
        
        self.samples.append(best_sample)

        return best_sample, z_ops

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