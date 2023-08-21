"""This module defines the abstract base class for all search algorithms."""

from abc import ABC, abstractmethod
import pickle
from typing import Callable, List, Tuple

import sample
import data_analysis
import const
import algo_args_type as algarty

import numpy as np


class SearchAlgo(ABC):
    """abstract base class for all search algorithms"""

    def __init__(self, algo_args: algarty.AlgoArgs):
        if algo_args is None:
            return  # empty instance to get the class name
        self.sig_generator = algo_args.sig_generator
        self.rand_args = algo_args.rand_args

        # unpack meta target's signal for backwards compatibility
        self.meta_target = algo_args.meta_target
        self.target = algo_args.meta_target.signal

        # parameters controlling runtime of all algorithms
        self.max_z_ops = algo_args.max_z_ops
        # k_samples is optional, and is inferred from z_ops if not provided

        # parameters applying to a subset of algorithms
        self.j_replace = algo_args.j_replace
        self.l_damp_prob = algo_args.l_damp_prob
        self.h_damp_fac = algo_args.h_damp_fac

        # parameters control how search runs, but don't influence results
        self.mp = algo_args.mp
        self.z_ops_callbacks = algo_args.z_ops_callbacks
        self.store_det_args = algo_args.store_det_args
        self.history = algo_args.history
        self.args_path = algo_args.args_path

        # optionally inferred parameter
        self.k_samples = (
            algo_args.k_samples if algo_args.k_samples is not None else self.infer_k_from_z()
        )

        self.algo_args = self.get_algo_args()  # for storage with algorithm results
        # this approach allows injecting from elsewhere
        # state
        self.all_samples: List[sample.Sample] = list()  # list of samples and results
        self.best_samples: List[
            Tuple[sample.Sample, int]
        ] = list()  # list of samples and z_ops for intermediate results
        self.z_ops: int = 0  # current number of operations

    def __str__(self) -> str:
        sig_gen_func = f"sig_generator=({self.sig_generator.__class__.__name__})\n"
        algo_args = f"algo_args=({self.get_algo_args()})\n"
        return sig_gen_func + algo_args

    def get_algo_args(self) -> algarty.AlgoArgs:
        """get the current set of AlgoArgs form the search module"""
        return algarty.AlgoArgs(
            self.sig_generator,
            self.rand_args,
            self.meta_target,
            self.max_z_ops,
            self.k_samples,
            self.j_replace,
            self.l_damp_prob,
            self.h_damp_fac,
            self.mp,
            self.z_ops_callbacks,
            self.store_det_args,
            self.history,
            self.args_path,
        )

    @staticmethod
    def gen_empty_sample() -> sample.Sample:
        """generate an empty oscillator ensemble"""
        return sample.Sample(None, None, None, 0, np.inf, list())

    def gen_zero_sample(self) -> sample.Sample:
        """generate a sample with all sensible fields set to 0"""
        samples = len(self.target)
        return sample.Sample(
            np.zeros((self.rand_args.n_osc, samples)),
            np.zeros(self.rand_args.n_osc),
            np.zeros(samples),
            0,
            data_analysis.compute_rmse(
                np.zeros(samples), self.target
            ),  # NOTE: np.inf makes more sense but; old solution kept for integrity of the experiment
            list(),
        )

    def pickle_samples(self, k: int) -> None:
        """pickle samples in RAM to disk in order to free RAM

        param:
            args_path:  where to store pickle
            k_samples:  total number of samples to draw
            k:          current k out of k_samples
        """
        if k == self.k_samples - 1 or k % const.SAMPLE_FLUSH_PERIOD == 0:
            with open(self.args_path, "ab") as f:
                pickle.dump(self.samples, f)
                self.samples = list()  # clear list

    def clear_state(self) -> None:
        """
        reset state

        1. delete file to which samples are written if it exists
        2. reset z_ops
        3. reset samples list
        """
        self.z_ops = 0
        self.samples = list()
        if self.history and self.args_path:
            self.args_path.unlink(missing_ok=True)

    def eval_z_ops_callback():
        """store a sample and current z_ops when the callback schedule says so"""
        NotImplemented
        # TODO: this is a bit difficult/ugly because we don't know which z_ops we will get exactly
        # therefore we must interpret the list of z_ops_callbacks as ranges
        # so we check whether the current z_ops is in a z_ops range
        # and then, if no sample has been added within this range, add the sample to the

    def manage_state(self, base_sample: sample.Sample, k: int) -> None:
        """save samples to RAM or file if desired"""
        # if self.z_ops_callbacks: self.eval_z_ops_callback()
        if self.history:
            self.samples.append(base_sample)
        if self.history and self.args_path:
            self.pickle_samples(k)  # TODO: could use len(self.samples)

    def set_first_point_to_zero(self) -> None:
        """
        set the first point in each sample to 0 and recompute rmse

        this may be desireable because signals will overlap at the first point
        this can also be prevented by using fully random phase-shifts
        """
        for s in self.samples:
            s.signal_sum[0] = 0  # set first point to 0
            s.rmse = data_analysis.compute_rmse(s.signal_sum, self.target)

    def gather_samples(self) -> tuple[sample.Sample, list]:
        """find the sample with the lowest root mean square error and return a list of all rmse"""
        best_sample = self.samples[0]
        rmse_li, rmse_norm_li = list(), list()

        for s in self.samples:
            rmse_li.append(s.rmse_sum)
            rmse_norm_li.append(s.rmse_norm)

            if s.rmse_sum < best_sample.rmse_sum:
                best_sample = s

        return best_sample, rmse_li, rmse_norm_li

    def stop_on_z_ops(self, verbose: bool = False) -> bool:
        """return true when an algorithm exceeds the maximum number of allowed operations"""
        if self.max_z_ops is None:
            return False
        if self.z_ops >= self.max_z_ops:
            if verbose:
                print(f"z_ops: {self.z_ops} > max_z_ops: {self.max_z_ops}")
            return True
        return False

    def accept_candidate_sample(self, rmse_base: float, rmse_temp: float) -> bool:
        """decide whether to accept a candidate sample"""
        if rmse_temp < rmse_base:
            return True
        return False

    def comp_samples(
        self, base_sample: sample.Sample, temp_sample: sample.Sample, *args, **kwargs
    ) -> sample.Sample:
        """compare two samples loss and greedily return the one with lower rmse"""
        if self.accept_candidate_sample(base_sample.rmse, temp_sample.rmse):
            return temp_sample
        return base_sample

    def draw_random_indices(self, j_replace: int) -> List[int]:
        """
        draw random indices pointing to an oscillator and weight for drawing a partial sample

        the index n points to the offset, where n is the number of oscillators
        """
        if self.mp:  # whether to use multiprocessing
            rng = np.random.default_rng()
            # choice uses an exclusive upper bound
            osc_to_replace = rng.choice(self.rand_args.n_osc + 1, size=j_replace, replace=False)
        else:
            osc_to_replace = const.RNG.choice(
                self.rand_args.n_osc + 1, size=j_replace, replace=False
            )
        return osc_to_replace

    def draw_weight_indices_or_offset(self, j_replace: int) -> Tuple[List[int], bool]:
        """draw random indices pointing to an oscillator and weight for drawing a partial sample

        compared to draw_random_indices, whether the offset is to be changed is passed as as bool instead of as index i=number_of_oscillators
        """
        # TODO: merge with separate_oscillators_from_offset in gen_signal.py
        osc_to_replace = self.draw_random_indices(j_replace)
        replace_offset = True if max(osc_to_replace) == self.rand_args.n_osc else False
        osc_to_replace = np.delete(osc_to_replace, np.where(osc_to_replace == self.rand_args.n_osc))
        return osc_to_replace, replace_offset

    def draw_sample(self) -> sample.Sample:
        """draw a sample and update z_ops"""
        self.z_ops += (
            self.rand_args.n_osc * 3 + 1
        )  # draw a new sample with n weighted oscillators (phase+frequency+gain --> 3) on each loop and an offset --> 1
        return self.sig_generator.draw_sample(self.rand_args, self.target, self.store_det_args)

    def draw_sample_weights(self, base_sample: sample.Sample):
        """update z_ops, draw new weights and offset for the sample and recompute metrics"""
        self.z_ops += self.rand_args.n_osc + 1  # n weights and 1 offset
        return self.sig_generator.draw_sample_weights(base_sample, self.rand_args, self.target)

    def draw_partial_sample(
        self, base_sample: sample.Sample, osc_to_replace: List[int]
    ) -> sample.Sample:
        """given a sample replace j oscillators and weights, update z_ops, recompute metrics"""
        self.z_ops += len(osc_to_replace) * 3  # len(osc_to_replace) == j_replace
        # draw j weighted oscillators (phase+frequency+gain --> 3)
        return self.sig_generator.draw_partial_sample(
            base_sample, self.rand_args, osc_to_replace, False, self.target, self.store_det_args
        )

    def draw_partial_sample_weights(
        self, base_sample: sample.Sample, osc_to_replace: List[int]
    ) -> sample.Sample:
        """given a sample replace j weights, update z_ops, recompute metrics"""
        self.z_ops += len(osc_to_replace)  # len(osc_to_replace) == j_replace
        return self.sig_generator.draw_partial_sample(
            base_sample, self.rand_args, osc_to_replace, True, self.target, self.store_det_args
        )

    def draw_weight_neighbor(
        self, base_sample: sample.Sample, osc_to_replace: List[int]
    ) -> sample.Sample:
        """given a sample replace j weights, update z_ops, recompute metrics

        weights are drawn from a neighborhood gaussian of the base sample instead of being drawn from the initial distribution
        """
        self.z_ops += len(osc_to_replace)
        return self.sig_generator.draw_weight_neighbor(
            base_sample, self.rand_args, osc_to_replace, self.target
        )

    def handle_mp(self, sup_func_kwargs: dict) -> None:
        """handle multi processing by modifying numpy the random number generator

        args:
            sup_func_kwargs: the kwargs of the calling function
        """
        # each process needs a unique seed
        if "mp" in sup_func_kwargs and sup_func_kwargs["mp"] == True:
            rng = np.random.default_rng(None)
            dist = self.rand_args.weight_dist.dist
            # __name__ is used to identify the function
            # this is incredibly ugly
            if isinstance(dist, Callable):
                # if dist is a uniform function, initialise it anew
                if dist.__name__ == rng.uniform.__name__:
                    # can't do
                    # dist = rng.uniform
                    self.rand_args.weight_dist.dist = rng.uniform
                elif dist.__name__ == rng.normal.__name__:
                    self.rand_args.weight_dist.dist = rng.normal

    @abstractmethod
    def init_best_sample(self) -> sample.Sample:
        """initialize best sample before first search loop"""
        raise NotImplementedError

    @abstractmethod
    def draw_temp_sample(self) -> sample.Sample:
        """draw a temporary sample to compare against the best sample and update z_ops"""
        raise NotImplementedError

    # TODO: incorporate offsets
    @abstractmethod
    def infer_k_from_z(self) -> int:
        """infer number of k-loops from a maximum number of operations z

        returns:
            k_samples: number of k-loops or temp_samples drawn
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, *args, **kwargs):  # *args needed to use with map(), not sure why
        """search for the best sample"""
        raise NotImplementedError
