"""This module implements an abstract base class for signal generators."""

import numpy as np

import copy
from abc import ABC, abstractmethod
from typing import Union, Tuple, List

import data_analysis
import gen_signal_args_types as party
import sample
import dist
import const


class SignalGenerator(ABC):
    """abstract base class for signal generators."""

    @staticmethod
    def draw_single_weight(rand_args: party.UnionRandArgs) -> float:
        """draw a single weight from the weight distribution"""
        return rand_args.weight_dist.draw()

    @staticmethod
    def draw_n_weights(rand_args: party.UnionRandArgs) -> np.ndarray:
        """draw n weights from the weight distribution"""
        return rand_args.weight_dist.draw_n()

    @staticmethod
    def draw_offset(rand_args: party.UnionRandArgs) -> float:
        """draw a single offset from the offset distribution"""
        offset = rand_args.offset_dist.draw()
        if offset is None:
            offset = 0
        return offset

    @staticmethod
    @abstractmethod
    def draw_params_random(rand_args: party.UnionRandArgs) -> party.UnionDetArgs:
        """draw a set of parameters from multiple distributions to initialize an oscillator

        args:
            rand_args: the random variables from which oscillators' parameters are drawn

        returns:
            a dataclass containing the deterministic parameters of an oscillator
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def draw_single_oscillator(rand_args: party.UnionRandArgs) -> np.ndarray:
        """draw a single oscillator from random variables

        args:
            rand_args: a dataclass containing deterministic and random variables from which oscillators are drawn
            store_det_args: whether to store the deterministic parameters underlying each oscillator

        returns:
            single_signal: a single oscillator
            det_args: the deterministic parameters underlying the oscillator
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def draw_n_oscillators(
        rand_args: party.UnionRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, List[Union[None, party.UnionDetArgs]]]:
        """compose a matrix of n-oscillators

        args:
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model

        returns:
            signal_matrix: a matrix of n-oscillators
            det_arg_li: a list of arguments used to generate each oscillator
        """
        raise NotImplementedError

    @staticmethod
    def get_adjacent_neighbors(osc_to_replace: np.ndarray, num_oscillators: int) -> List[int]:
        """add the two adjacent neighbors to a list of oscillators"""
        if len(osc_to_replace) < 1:
            return osc_to_replace

        neighbors = list(osc_to_replace)
        lower_neighbor = int(np.min(osc_to_replace) - 1)
        upper_neighbor = int(np.max(osc_to_replace) + 1)
        if lower_neighbor >= 0:
            neighbors = [lower_neighbor] + neighbors
        if upper_neighbor < num_oscillators:
            neighbors = neighbors + [upper_neighbor]
        return np.array(neighbors)

    @staticmethod
    def update_weight_distribution(
        base_sample: sample.Sample, temp_args: party.UnionRandArgs, neighborhood: List[int]
    ) -> party.UnionRandArgs:
        """update the weight distribution to draw from a gaussian over the neighborhood"""
        temp_args.n_osc = len(neighborhood)
        mean_weight = np.mean(base_sample.weights[neighborhood])
        stddev = np.std(base_sample.weights)
        temp_args.weight_dist = dist.WeightDist(
            const.RNG.normal, loc=mean_weight, scale=stddev, n=len(neighborhood)
        )
        return temp_args

    def draw_sample(
        self,
        rand_args: party.UnionRandArgs,
        target: Union[None, np.ndarray] = None,
        store_det_args: bool = False,
    ) -> sample.Sample:
        """draw a sample from scratch and compute available metrics

        args:
            rand_args: a dataclass containing deterministic and random variables from which oscillators are drawn
            target: a target signal to compare the generated signal to
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model

        returns:
            a sample containing the generated signal
            sample contains matrix, weights, weighted sum, offset, rmse and deterministic arguments underlying the oscillators in the model
        """
        signal_matrix, det_args = self.draw_n_oscillators(rand_args, store_det_args)
        weights = self.draw_n_weights(rand_args)
        offset = self.draw_offset(rand_args)
        weighted_sum = sample.Sample.compute_weighted_sum(signal_matrix, weights, offset)
        rmse = None
        if target is not None:
            rmse = data_analysis.compute_rmse(weighted_sum, target)
        return sample.Sample(signal_matrix, weights, weighted_sum, offset, rmse, det_args)

    def draw_sample_weights(
        self,
        base_sample: sample.Sample,
        rand_args: party.UnionRandArgs,
        target: Union[None, np.ndarray] = None,
    ) -> sample.Sample:
        """replace only the weights and offset of a sample

        args:
            base_sample: the sample to copy
            rand_args: a dataclass containing deterministic and random variables from which the weights are drawn

        returns:
            the base sample with new weights and re-computed metrics
        """
        updated_sample = copy.deepcopy(base_sample)
        updated_sample.weights = self.draw_n_weights(rand_args)
        updated_sample.offset = self.draw_offset(rand_args)
        updated_sample.weighted_sum = sample.Sample.compute_weighted_sum(
            updated_sample.signal_matrix, updated_sample.weights, updated_sample.offset
        )
        updated_sample.rmse = None
        if target is not None:
            updated_sample.rmse = data_analysis.compute_rmse(updated_sample.weighted_sum, target)
        return updated_sample

    @staticmethod
    def separate_oscillators_from_offset(
        osc_to_replace: np.ndarray, n_oscillators: int
    ) -> Tuple[np.ndarray, bool]:
        """evaluate whether to replace the offset and clean the list of oscillators to replace

        args:
            osc_to_replace: list of indices of oscillators to be replaced, index i=number_of_oscillators points to the offset
            n_oscillators: number of oscillators in a model (sample)

        returns:
            osc_to_replace: list of indices of oscillators with the offset removed
            replace_offset: whether to replace the offset
        """
        replace_offset = True if np.max(osc_to_replace) == n_oscillators else False
        osc_to_replace = np.delete(osc_to_replace, np.where(osc_to_replace == n_oscillators))
        return osc_to_replace, replace_offset

    # TODO: split into two functions: draw_partial_sample and draw_partial_sample_weights
    def draw_partial_sample(
        self,
        base_sample: sample.Sample,
        rand_args: party.UnionRandArgs,
        osc_to_replace: List[int],
        weight_mode: bool,
        target: Union[None, np.ndarray] = None,
        store_det_args: bool = False,
    ) -> sample.Sample:
        """
        take a base sample and replace j oscillators and weights.

        this function generalizes over draw_sample;
        draw_sample is kept because it's more readable

        args:
            base_sample: a sample to be modified
            rand_args: a dataclass containing deterministic and random variables from which oscillators are drawn
            osc_to_replace: a list of indices of oscillators to be replaced, j_replace == len(osc_to_replace)
            weight_mode: when true only replace weights, when false, replace weights and oscillators
            indices: indices at which to replace oscillators and weights
            target: a target signal to compare the generated signal to
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model

        returns:
            a sample containing the generated signal
        """
        osc_to_replace, replace_offset = self.separate_oscillators_from_offset(
            osc_to_replace, rand_args.n_osc
        )

        new_sample = copy.deepcopy(base_sample)  # copy the base sample
        temp_args = copy.deepcopy(rand_args)  # copy to avoid side effects

        # update the underlying distributions by setting n (number of oscillators) to j (number of oscillators to replace)
        temp_args.n_osc = len(osc_to_replace)  # len(osc_to_replace) == j_replace
        temp_args.weight_dist.n = len(osc_to_replace)

        # draw a new set of oscillators
        if not weight_mode:
            partial_signal_matrix, det_args = self.draw_n_oscillators(temp_args, store_det_args)
            new_sample.signal_matrix[osc_to_replace] = partial_signal_matrix

            # keep track of the parameters underlying the oscillators
            if det_args is not None:
                for osc_index, det_arg in zip(osc_to_replace, det_args):
                    new_sample.det_args[osc_index] = det_arg

        # draw a new set of weights
        partial_weights = self.draw_n_weights(temp_args)
        new_sample.weights[osc_to_replace] = partial_weights

        # draw new offset
        if replace_offset:
            new_sample.offset = self.draw_offset(temp_args)

        new_sample.weighted_sum = sample.Sample.compute_weighted_sum(
            new_sample.signal_matrix, new_sample.weights, new_sample.offset
        )
        new_sample.rmse = None
        if target is not None:
            new_sample.rmse = data_analysis.compute_rmse(new_sample.weighted_sum, target)

        return new_sample

    def draw_weight_neighbor(
        self,
        base_sample: sample.Sample,
        rand_args: party.UnionRandArgs,
        osc_to_replace: List[int],
        target: np.ndarray,
    ) -> sample.Sample:
        """take a base sample and replace j weights.

        compared to draw_partial_sample, this function draws weights from a distribution centered around the
        mean weight of the oscillators to be reweighted.
        """

        def update_weights(
            new_sample: sample.Sample, temp_args: party.UnionRandArgs, neighborhood: List[int]
        ) -> sample.Sample:
            """draw new weights and update a sample"""
            partial_weights = self.draw_n_weights(temp_args)
            new_sample.weights[neighborhood] = partial_weights
            return new_sample

        # separate offset from weights
        osc_to_replace, replace_offset = self.separate_oscillators_from_offset(
            osc_to_replace, rand_args.n_osc
        )

        # avoid side effects
        new_sample = copy.deepcopy(base_sample)
        temp_args = copy.deepcopy(rand_args)

        neighborhood = self.get_adjacent_neighbors(osc_to_replace, temp_args.n_osc)
        temp_args = self.update_weight_distribution(new_sample, temp_args, neighborhood)
        new_sample = update_weights(new_sample, temp_args, neighborhood)

        # draw new offset
        if replace_offset:
            new_sample.offset = self.draw_offset(temp_args)

        new_sample.weighted_sum = sample.Sample.compute_weighted_sum(
            new_sample.signal_matrix, new_sample.weights, new_sample.offset
        )
        new_sample.rmse = None
        if target is not None:
            new_sample.rmse = data_analysis.compute_rmse(new_sample.weighted_sum, target)

        return new_sample
