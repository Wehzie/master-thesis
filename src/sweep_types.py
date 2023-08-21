"""This module defines valid dependent variables for an experiment (sweep)."""

from abc import ABC
from dataclasses import dataclass
from typing import Final, List, Union
import dist
import algo_args_bundle
import mask_type
import meta_target
import gen_signal_args_types as party
import gen_signal


@dataclass
class AlgoSweep:
    """
    Repeat experiments over multiple algorithms.

    args:
        algo_with_args: list of algorithms with their arguments
        m_averages: number of averages for each experimental configuration
        algo_masks: masks to subset the algorithms for analysis; doesn't affect the experiment
    """

    algo_with_args: List[algo_args_bundle.AlgoWithArgs]
    m_averages: int
    algo_masks: Union[List[mask_type.ExperimentMask], None] = None


# TODO: maybe use this instead of AlgoSweep
@dataclass
class AlgoSweepBundle:
    """Bundle an AlgoSweep with additional information."""

    name: str
    description: str
    dep_var: None
    target: meta_target.MetaTarget
    sig_generator: None
    generator_args: None
    algo_with_args: None
    max_z_ops: int
    m_averages: int
    algo_masks: None


@dataclass
class ConstTimeSweep(ABC):
    """
    Abstract class for experiments where the time complexity between experiments is constant.

    For example, increasing the frequency doesn't increase the time complexity of the experiment.
    """

    pass


@dataclass
class FreqSweep(ConstTimeSweep):
    """A list of frequency distributions from which to sample the frequency of the oscillators."""

    freq_dist: List[dist.Dist]


@dataclass
class ResistorSweep(ConstTimeSweep):
    """
    A list of resistor distributions from which to sample the resistance of the RC-circuit.

    Resistance controls the frequency of the oscillators.
    """

    r_dist: List[dist.Dist]


@dataclass
class AmplitudeSweep(ConstTimeSweep):
    """A list of amplitude distributions from which to sample the amplitude of the oscillator signals."""

    amplitude: List[float]


@dataclass
class WeightSweep(ConstTimeSweep):
    """A list of weight distributions from which to sample the weighting or gain of the oscillator signals."""

    weight_dist: List[dist.WeightDist]


@dataclass
class PhaseSweep(ConstTimeSweep):
    """A list of phase distributions from which to sample the phase of the oscillator signals."""

    phase_dist: List[dist.Dist]


@dataclass
class OffsetSweep(ConstTimeSweep):
    """A list of offset distributions from which to sample the offset of the oscillator signals."""

    offset_dist: List[dist.Dist]


@dataclass
class ExpoTimeSweep(ABC):
    """
    This abstract class specifies experiments where time complexity between experiments is worse then constant.

    For example, increasing the number of oscillators increases the duration and memory requirements of the experiment.

    args:
        filename: filename for saving the results
        iv_identifier: independent variable identifier, must match a corresponding field in the RandSignalArgs class
        iv_description: description of the independent variable
        val_schedule: values of the independent variable
        dv_identifier: dependent variable identifier; not yet implemented; must match a corresponding field in a dataframe returned by the Experimenteur class
    """

    NotImplemented
    # TODO: using fixed field names I can replace
    #   for val_schedule in fields(sweep_args)
    #       for awa in algo_sweep.algo_with_args:
    #           for val in getattr(sweep_args, val_schedule):
    #               ...
    # in the experimenteur class module with
    #   for awa in algo_sweep.algo_with_args:
    #       for val in sweep_args.val_schedule:
    #           ...

    # filename: str                       # filename for saving the results
    # iv_identifier: str                  # mu
    # iv_description: str                 # description of the independent variable
    # val_schedule: List                  # values of the independent variable
    # dv_identifier: str                        # type of the dependent variable


@dataclass
class NOscSweep(ExpoTimeSweep):
    """Specify varying numbers of oscillators to compare in an experiment."""

    n_osc: List[int]


@dataclass
class ZOpsSweep(ExpoTimeSweep):
    """Specify varying numbers of maximum perturbations (max-z-ops) to use in an experiment."""

    max_z_ops: List[int]


@dataclass
class NumSamplesSweep(ExpoTimeSweep):
    """
    Specify varying numbers of target samples to use in an experiment.

    This approximates the idea of varying the duration of the target signal.
    """

    samples: List[float]


@dataclass
class DurationSweep(ExpoTimeSweep):
    """Specify varying durations of target samples to use in an experiment."""

    duration: List[float]


@dataclass
class TargetSweep(ExpoTimeSweep):
    """
    Specify varying target signals to use in an experiment.

    args:
        description: text description of the experiment
        targets: list of target signals against which to compare the algorithms
        rand_args: a set of parameters used by the signal generator to generate the target signals
        signal_generator: a signal generator
        max_z_ops: maximum number of perturbations to use in the experiment
        m_averages: number of averages for each experimental configuration
    """

    description: str
    targets: List[meta_target.MetaTarget]
    rand_args: party.UnionRandArgs
    signal_generator: gen_signal.SignalGenerator
    max_z_ops: int
    m_averages: int


@dataclass
class SweepBundle:
    """Bundle an algorithm sweep with sweeps of secondary independent variables"""

    # metadata, mainly for convenience
    # the values of these fields should not change between experiments
    # for example, max_z_ops reflects the default values
    # while z_ops_sweep reflects the values to be explored
    description: str
    signal_generator: Final[gen_signal.SignalGenerator]
    generator_args: Final[party.UnionRandArgs]
    max_z_ops: Final[int]
    m_averages: Final[int]

    # primary dependent variable shared by all experiments
    algo_sweep: Final[AlgoSweep]

    # secondary dependent variables
    # exponential time sweeps
    target_sweep: Final[TargetSweep]
    n_osc_sweep: Final[NOscSweep]
    z_ops_sweep: Final[ZOpsSweep]
    duration_sweep: Union[DurationSweep, None]  # extrapolation to longer duration

    # constant time sweeps
    weight_sweep: Final[WeightSweep]
    phase_sweep: Final[PhaseSweep]
    offset_sweep: Final[OffsetSweep]


@dataclass
class PythonSweepBundle(SweepBundle):
    """Bundle an AlgoSweep with a sweep of secondary independent variables for the Python signal generator."""

    # constant time sweeps
    freq_sweep_from_zero: Final[FreqSweep]
    freq_sweep_around_vo2: Final[FreqSweep]
    amplitude_sweep: Final[AmplitudeSweep]
    num_samples_sweep: Final[NumSamplesSweep]


@dataclass
class HybridSweepBundle(SweepBundle):
    """Bundle an AlgoSweep with a set of secondary independent variables for the Hybrid/Spipy signal generator."""

    # constant time sweeps
    resistor_sweep: Final[FreqSweep]  # manipulate netlist generation, R value
    target_freq_sweep: Final[TargetSweep]


UnionSweepBundle = Union[PythonSweepBundle, HybridSweepBundle]
