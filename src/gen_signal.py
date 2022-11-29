import numpy as np

import copy
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Callable

import data_analysis
import param_types as party
import sample



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
        return rand_args.offset_dist.draw()

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
    def draw_n_oscillators(rand_args: party.UnionRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, List[Union[None, party.UnionDetArgs]]]:
        """compose a matrix of n-oscillators

        args:
            store_det_args: whether to store the deterministic parameters underlying each oscillator in a model

        returns:
            signal_matrix: a matrix of n-oscillators
            det_arg_li: a list of arguments used to generate each oscillator
        """
        raise NotImplementedError

    def draw_sample(self, rand_args: party.UnionRandArgs, target: Union[None, np.ndarray] = None, store_det_args: bool = False) -> sample.Sample:
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
    
    def draw_partial_sample(self, base_sample: sample.Sample, rand_args: party.UnionRandArgs,
    osc_to_replace: List[int], weight_mode: bool,
    target: Union[None, np.ndarray] = None, store_det_args: bool = False,
    ) -> sample.Sample:
        """take a base sample and replace j oscillators and weights.
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
        new_sample = copy.deepcopy(base_sample) # copy the base sample
        temp_args = copy.deepcopy(rand_args) # copy to avoid side effects
        
        # update the underlying distributions by setting n (number of oscillators) to j (number of oscillators to replace)
        temp_args.n_osc = len(osc_to_replace) # len(osc_to_replace) == j_replace
        temp_args.weight_dist.n = len(osc_to_replace)

        # draw a new set of oscillators
        if not weight_mode:
            partial_signal_matrix, det_args = self.draw_n_oscillators(temp_args, store_det_args)
            new_sample.signal_matrix[osc_to_replace] = partial_signal_matrix

            # keep track of the parameters underlying the oscillators
            if det_args is not None:
                for (osc_index, det_arg) in zip(osc_to_replace, det_args):
                    new_sample.det_args[osc_index] = det_arg

        # draw a new set of weights
        partial_weights = self.draw_n_weights(temp_args)
        new_sample.weights[osc_to_replace] = partial_weights

        new_sample.weighted_sum = sample.Sample.compute_weighted_sum(new_sample.signal_matrix,
            new_sample.weights, new_sample.offset)
        new_sample.rmse = None
        if target is not None:
            new_sample.rmse = data_analysis.compute_rmse(new_sample.weighted_sum, target)

        return new_sample

    def draw_sample_weights(self, base_sample: sample.Sample, rand_args: party.UnionRandArgs, target: Union[None, np.ndarray] = None) -> sample.Sample:
        """return the base sample with all weights replaced and recomputed metrics
        
        args:
            base_sample: the sample to copy
            rand_args: a dataclass containing deterministic and random variables from which the weights are drawn
        
        returns:
            the base sample with new weights and re-computed metrics
        """
        updated_sample = copy.deepcopy(base_sample)
        updated_sample.weights = self.draw_n_weights(rand_args)
        # TODO: draw a new offset! Also fix the algorithms in that way
        # updated_sample.offset = gen_signal.SignalGenerator.draw_offset(rand_args)
        updated_sample.weighted_sum = sample.Sample.compute_weighted_sum(updated_sample.signal_matrix, updated_sample.weights, updated_sample.offset)
        updated_sample.rmse = None
        if target is not None:
            updated_sample.rmse = data_analysis.compute_rmse(updated_sample.weighted_sum, target)   
        return updated_sample