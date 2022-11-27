import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Callable

import param_types as party
import sample

UnionRandArgs = Union[party.PythonSignalRandArgs, party.SpiceSumRandArgs]
UnionDetArgs = Union[party.PythonSignalDetArgs, party.SpiceSingleDetArgs, party.SpiceSumDetArgs]

class SignalGenerator(ABC):
    """Abstract base class for signal generators."""

    @staticmethod
    @abstractmethod
    def draw_single_oscillator(rand_args: UnionRandArgs) -> np.ndarray:
        """Draw a single oscillator signal."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def draw_n_oscillators(rand_args: UnionRandArgs, store_det_args: bool = False
    ) -> Tuple[np.ndarray, List[Union[None, UnionDetArgs]]]:
        """Draw n oscillators and return a matrix with one signal per row."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def draw_sample(rand_args: UnionRandArgs) -> sample.Sample:
        """Draw a sample from the signal generator."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def draw_params_random(args: UnionRandArgs) -> UnionDetArgs:
        """draw randomly from parameter pool"""
        raise NotImplementedError

    @staticmethod
    def draw_single_weight(rand_args: UnionRandArgs) -> float:
        return rand_args.weight_dist.draw()

    @staticmethod
    def draw_n_weights(rand_args: UnionRandArgs) -> np.ndarray:
        return rand_args.weight_dist.draw_n()

    @staticmethod
    def draw_offset(rand_args: UnionRandArgs) -> float:
        return rand_args.offset_dist.draw()