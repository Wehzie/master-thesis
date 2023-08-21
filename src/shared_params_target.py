"""
This module initializes synthetic target signals with here-defined parameters.

Furthermore, it provides paths for loading real world signals from file.
"""

from pathlib import Path
from enum import Enum
from typing import List, Union

import meta_target
import gen_signal_args_types as party

#### #### #### #### AUDIO FILE PATHS #### #### #### ####


class DevSet(Enum):  # used for development of the pipeline
    """development set of targets that can be loaded from file"""

    MAGPIE = Path("resources/magpie. 35k, mono, 8-bit, 11025 Hz, 3.3 seconds.wav")
    YES = Path("resources/yes-5.wav")


class TestSet(Enum):  # held out until the end of development
    """test set of targets that can be loaded from file"""

    BELLBIRD = Path("resources/white_bellbird.wav")
    OKAY = Path("resources/okay-7.wav")
    # HIGH-RES IDEA https://www.mitsue.co.jp/english/service/audio_and_video/audio_production/high_resolution_narration.html


#### #### #### #### PRODUCTION SYNTHETIC TARGETS #### #### #### ####


def build_production_targets(
    duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None
) -> List[meta_target.MetaTarget]:
    """build a list of targets for production"""
    assert None in [
        sampling_rate,
        samples,
    ], "either sampling rate or number of samples should be specified, but not both"

    single_frequency_arg_targets = []
    for Target in [
        meta_target.SineTarget,
        meta_target.TriangleTarget,
        meta_target.SawtoothTarget,
        meta_target.InverseSawtoothTarget,
        meta_target.SquareTarget,
    ]:
        for freq in [1e3, 1e4, 1e5, 1e6]:
            single_frequency_arg_targets.append(
                Target(duration, sampling_rate, samples=samples, freq=freq)
            )

    beat_targets = [
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e3),
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e4),
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e5),
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e6),
    ]

    difficult_targets = [
        meta_target.ChirpTarget(
            duration, sampling_rate, samples=samples, start_freq=1, stop_freq=1e6
        ),
        meta_target.DampChirpTarget(
            duration, sampling_rate, samples=samples, start_freq=1, stop_freq=1e6
        ),
        meta_target.SmoothGaussianNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.SmoothUniformNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.GaussianNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.UniformNoiseTarget(duration, sampling_rate, samples=samples),
    ]

    production_targets = single_frequency_arg_targets + beat_targets + difficult_targets
    return production_targets


def build_target_freq_sweep(
    duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None
) -> List[meta_target.MetaTarget]:
    target_freq_sweep = [
        meta_target.SineTarget(duration, sampling_rate, samples=samples, freq=1, name="sine 1 Hz"),
        meta_target.SineTarget(
            duration, sampling_rate, samples=samples, freq=1e1, name="sine 10 Hz"
        ),
        meta_target.SineTarget(
            duration, sampling_rate, samples=samples, freq=1e2, name="sine 100 Hz"
        ),
        meta_target.SineTarget(
            duration, sampling_rate, samples=samples, freq=1e3, name="sine 1e3 Hz"
        ),
        meta_target.SineTarget(
            duration, sampling_rate, samples=samples, freq=1e4, name="sine 1e4 Hz"
        ),
        meta_target.SineTarget(
            duration, sampling_rate, samples=samples, freq=1e5, name="sine 1e5 Hz"
        ),
        meta_target.SineTarget(
            duration, sampling_rate, samples=samples, freq=1e6, name="sine 1e6 Hz"
        ),
        meta_target.DampChirpTarget(
            duration,
            sampling_rate,
            samples=samples,
            start_freq=1,
            stop_freq=1e1,
            name="d. chirp 1 to 10 Hz",
        ),
        meta_target.DampChirpTarget(
            duration,
            sampling_rate,
            samples=samples,
            start_freq=1e1,
            stop_freq=1e2,
            name="d. chirp 10 to 100 Hz",
        ),
        meta_target.DampChirpTarget(
            duration,
            sampling_rate,
            samples=samples,
            start_freq=1e2,
            stop_freq=1e3,
            name="d. chirp 100 to 1e3 Hz",
        ),
        meta_target.DampChirpTarget(
            duration,
            sampling_rate,
            samples=samples,
            start_freq=1e3,
            stop_freq=1e4,
            name="d. chirp 1e3 to 1e4 Hz",
        ),
        meta_target.DampChirpTarget(
            duration,
            sampling_rate,
            samples=samples,
            start_freq=1e4,
            stop_freq=1e5,
            name="d. chirp 1e4 to 1e5 Hz",
        ),
        meta_target.DampChirpTarget(
            duration,
            sampling_rate,
            samples=samples,
            start_freq=1e5,
            stop_freq=1e6,
            name="d. chirp 1e5 to 1e6 Hz",
        ),
    ]
    return target_freq_sweep


#### #### #### #### TEST SYNTHETIC TARGETS #### #### #### ####


def build_test_targets(
    duration: float,
    sampling_rate: Union[int, None] = None,
    samples: Union[int, None] = None,
    synth_freq: float = 1e4,
) -> List[meta_target.MetaTarget]:
    """build a list of targets for testing the pipeline"""
    assert None in [
        sampling_rate,
        samples,
    ], "either sampling rate or number of samples should be specified, but not both"

    single_frequency_arg_targets_test = []
    for Target in [
        meta_target.SineTarget,
        meta_target.TriangleTarget,
        meta_target.SawtoothTarget,
        meta_target.InverseSawtoothTarget,
        meta_target.SquareTarget,
    ]:
        single_frequency_arg_targets_test.append(
            Target(duration, sampling_rate, samples=samples, freq=synth_freq)
        )

    test_beat_target = [
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=synth_freq)
    ]

    difficult_targets = [
        meta_target.ChirpTarget(
            duration, sampling_rate, samples=samples, start_freq=1, stop_freq=synth_freq
        ),
        meta_target.DampChirpTarget(
            duration, sampling_rate, samples=samples, start_freq=1, stop_freq=synth_freq
        ),
        meta_target.SmoothGaussianNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.SmoothUniformNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.GaussianNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.UniformNoiseTarget(duration, sampling_rate, samples=samples),
    ]

    test_targets = single_frequency_arg_targets_test + test_beat_target + difficult_targets
    return test_targets


def select_target_by_string(
    selector: str, generator_args: party.UnionRandArgs, synth_freq: float, duration: float = None
) -> meta_target.MetaTargetSample:
    """
    Select a target function by its name.

    args:
        generator_args: (rand args) the arguments required by the signal generator
        synth_freq: the frequency of the synthesized signal
        selector: the string to select the target sample
    """
    amplitude = 1
    if isinstance(generator_args, party.PythonSignalRandArgs):
        duration = duration
        sampling_rate = None
        samples = generator_args.samples
    elif isinstance(generator_args, party.SpiceSumRandArgs):
        duration = generator_args.get_duration()
        sampling_rate = generator_args.get_sampling_rate()
        samples = None
    else:
        raise ValueError("unrecognized signal generator args type")

    if selector == "magpie":
        if isinstance(generator_args, party.PythonSignalRandArgs):
            m_target = meta_target.MetaTargetSample(generator_args, "magpie", DevSet.MAGPIE.value)
        elif isinstance(generator_args, party.SpiceSumRandArgs):
            m_target = meta_target.MetaTargetTime(generator_args, "magpie", DevSet.MAGPIE.value)
    elif selector == "human_yes":
        if isinstance(generator_args, party.PythonSignalRandArgs):
            m_target = meta_target.MetaTargetSample(generator_args, "human_yes", DevSet.YES.value)
        elif isinstance(generator_args, party.SpiceSumRandArgs):
            m_target = meta_target.MetaTargetTime(generator_args, "human_yes", DevSet.YES.value)
    elif selector == "bellbird":
        if isinstance(generator_args, party.PythonSignalRandArgs):
            m_target = meta_target.MetaTargetSample(
                generator_args, "bellbird", TestSet.BELLBIRD.value
            )
        elif isinstance(generator_args, party.SpiceSumRandArgs):
            m_target = meta_target.MetaTargetTime(
                generator_args, "bellbird", TestSet.BELLBIRD.value
            )
    elif selector == "human_okay":
        if isinstance(generator_args, party.PythonSignalRandArgs):
            m_target = meta_target.MetaTargetSample(
                generator_args, "human_okay", TestSet.OKAY.value
            )
        elif isinstance(generator_args, party.SpiceSumRandArgs):
            m_target = meta_target.MetaTargetTime(generator_args, "human_okay", TestSet.OKAY.value)
    elif selector == "sine":
        m_target = meta_target.SineTarget(duration, sampling_rate, samples, synth_freq, amplitude)
    elif selector == "triangle":
        m_target = meta_target.TriangleTarget(
            duration, sampling_rate, samples, synth_freq, amplitude
        )
    elif selector == "square":
        m_target = meta_target.SquareTarget(duration, sampling_rate, samples, synth_freq, amplitude)
    elif selector == "sawtooth":
        m_target = meta_target.SawtoothTarget(
            duration, sampling_rate, samples, synth_freq, amplitude
        )
    elif selector == "inverse_sawtooth":
        m_target = meta_target.InverseSawtoothTarget(
            duration, sampling_rate, samples, synth_freq, amplitude
        )
    elif selector == "chirp":
        m_target = meta_target.ChirpTarget(
            duration,
            sampling_rate,
            samples,
            start_freq=1,
            stop_freq=synth_freq,
            amplitude=amplitude,
        )
    elif selector == "beat":
        m_target = meta_target.BeatTarget(
            duration, sampling_rate, samples, base_freq=synth_freq, amplitude=amplitude
        )
    elif selector == "damp_chirp":
        m_target = meta_target.DampChirpTarget(
            duration,
            sampling_rate,
            samples,
            start_freq=1,
            stop_freq=synth_freq,
            amplitude=amplitude,
        )
    elif selector == "smooth_gauss":
        m_target = meta_target.SmoothGaussianNoiseTarget(
            duration, sampling_rate, samples, amplitude
        )
    elif selector == "smooth_uniform":
        m_target = meta_target.SmoothUniformNoiseTarget(duration, sampling_rate, samples, amplitude)
    elif selector == "gauss_noise":
        m_target = meta_target.GaussianNoiseTarget(duration, sampling_rate, samples, amplitude)
    elif selector == "uniform_noise":
        m_target = meta_target.UniformNoiseTarget(duration, sampling_rate, samples, amplitude)
    else:
        raise ValueError(f"Target {selector} not recognized.")
    return m_target
