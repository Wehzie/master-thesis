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

import numpy as np
from tqdm import tqdm

class MCOneShot(algo.SearchAlgo):
    """monte carlo algorithm for samples consisting of independent oscillators"""

    @data_analysis.print_time
    def search(self) -> Tuple[sample.Sample, int]:
        """generate k-signals which are a sum of n-oscillators
        on each iteration draw a new full model (matrix of n-oscillators)
        
        params:
            k_samples: number of times to draw a matrix
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model
            history: whether to store all generated samples, may run out of RAM
            args_path: when not none, write history to disk instead of RAM at specified path
        """
        self.clear_state()

        best_sample = self.init_best_sample()
        for k in tqdm(range(self.k_samples)):
            temp_sample = self.draw_temp_sample(best_sample)
            self.manage_state(temp_sample, k)
            best_sample = self.comp_samples(best_sample, temp_sample)
            if self.eval_z_ops(): return best_sample, self.z_ops

        return best_sample, self.z_ops

class MCExploit(algo.SearchAlgo):
    """monte carlo algorithm exploiting a single sample by iterative re-draws"""
    def init_random_exploit(self, zero_model: bool) -> sample.Sample:
        if zero_model:
            signal_matrix, signal_args = np.zeros((self.rand_args.n_osc, self.rand_args.samples)), list()
        else:
            signal_matrix, signal_args = gen_signal_python.sum_atomic_signals(self.rand_args)
        base_sample = sample.Sample(signal_matrix, None, None, 0, None, signal_args)
        base_sample.update(self.target)
        return base_sample

    def draw_exploit_candidate(self, base_sample: sample.Sample, mod_args: party.PythonSignalRandArgs, store_det_args: bool, history: bool):
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

    def compute_z_ops(self, k_samples: int, zero_init: bool, j_exploits: int) -> int:
        if zero_init:
            z_ops = k_samples * j_exploits
        else:
            z_ops = k_samples * (self.rand_args.n_osc + j_exploits)
        return z_ops

    def random_exploit(self, k_samples: int, j_exploits: int, zero_model: bool = False) -> None:
        """generate an ensemble of n-oscillators
        reduce loss by iteratively replacing oscillators in the ensemble
        
        param:
            j_exploits: number of re-drawn oscillators per model
            zero_model: if True, start out with a model of zero-oscillators
        """
        # TODO: hacky solution to draw l-oscillators from sum_atomic signals
        mod_args = copy.deepcopy(self.rand_args)
        mod_args.n_osc = 1

        best_sample = sample.Sample(None, None, None, 0, np.inf, list())
        for _ in tqdm(range(k_samples)):
            # generate initial model
            
            base_sample = self.init_random_exploit(zero_model)
            for _ in range(j_exploits):
                # draw 1 oscillator
                signal, _ = gen_signal_python.sum_atomic_signals(mod_args)
                signal = signal.flatten()

                # replace random oscillator in the model
                i = rng.integers(0, self.rand_args.n_osc)
                # TODO: debug whether this works as expected or whether deepcopy is needed
                temp_sum = np.sum(base_sample.signal_matrix[0:i,:], axis=0) + signal + np.sum(base_sample.signal_matrix[1+i:,:], axis=0)
                temp_rmse = data_analysis.compute_rmse(temp_sum, self.target)

                # evaluate replacement
                if temp_rmse < base_sample.rmse:
                    base_sample.signal_matrix[i,:] = signal
                    base_sample.rmse = temp_rmse
                    base_sample.signal_sum = temp_sum

            if base_sample.rmse < best_sample.rmse:
                best_sample = base_sample

        self.samples.append(best_sample)
        
        return best_sample, self.compute_z_ops(k_samples, zero_model, j_exploits)
        
    def random_weight_exploit(self,
        k_samples: int,
        j_exploits: int,
        weight_init: str,
        store_det_args: bool = False,
        history: bool = False,
        args_path: Path = None) -> Tuple[sample.Sample, int]:
        """generate an ensemble of n-oscillators and a vector of weights
        reduce loss by iteratively replacing weights
        
        param:
            j_exploits: number of re-drawn oscillators per model
        """
        # TODO: hacky solution to draw l-oscillators from sum_atomic signals

        best_sample = sample.Sample(None, None, None, 0, np.inf, list())
        for _ in range(k_samples):

            base_sample = self.init_las_vegas(weight_init, store_det_args)
            for _ in range(j_exploits):
                i = rng.integers(0, self.rand_args.n_osc)
                temp_sample = self.draw_las_vegas_weight_candidate(base_sample, i)
                base_sample, _, _ = self.eval_las_vegas(temp_sample, base_sample, 0, 0, 0)

            if base_sample.rmse < best_sample.rmse:
                best_sample = base_sample

        best_sample.signal_matrix = (best_sample.signal_matrix.T * best_sample.weights).T # TODO: change sample type
        self.samples.append(best_sample)
        zero_init = True if weight_init == "zeros" else False
        return best_sample, self.compute_z_ops(k_samples, zero_init, j_exploits)
        