"""
This module initializes Synthetic target signals with here-defined parameters.
Furthermore, it provides paths for loading real world signals from file.
"""

from pathlib import Path
from enum import Enum
from typing import List, Union

import meta_target

#### #### #### #### AUDIO FILE PATHS #### #### #### #### 

class DevSet(Enum): # used for development of the pipeline
    """development set of targets that can be loaded from file"""
    MAGPIE = Path("resources/magpie. 35k, mono, 8-bit, 11025 Hz, 3.3 seconds.wav")
    YES = Path("resources/yes-5.wav")

class TestSet(Enum): # held out until the end of development
    """test set of targets that can be loaded from file"""
    BELLBIRD = Path("resources/white_bellbird.wav")
    OKAY = Path("resources/okay-7.wav")

#### #### #### #### PRODUCTION SYNTHETIC TARGETS #### #### #### #### 

def build_production_targets(duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None) -> List[meta_target.MetaTarget]:
    """builds a list of targets for production"""
    assert None in [sampling_rate, samples], "either sampling rate or number of samples should be specified, but not both"

    single_frequency_arg_targets = []
    for Target in [meta_target.SineTarget, meta_target.TriangleTarget, meta_target.SawtoothTarget, meta_target.InverseSawtoothTarget, meta_target.SquareTarget]:
        for freq in [1e1, 1e2, 1e3, 1e4, 1e5]:
            single_frequency_arg_targets.append(Target(duration, sampling_rate, samples=samples, freq=freq))

    beat_targets = [
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e1),
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e2),
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e3),
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e4),
        meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=1e5),
    ]

    difficult_targets = [
        meta_target.ChirpTarget(duration, sampling_rate, samples=samples, start_freq=1, stop_freq=1e5),
        meta_target.DampChirpTarget(duration, sampling_rate, samples=samples, start_freq=1, stop_freq=1e5),
        meta_target.SmoothGaussianNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.SmoothUniformNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.GaussianNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.UniformNoiseTarget(duration, sampling_rate, samples=samples),
    ]

    production_targets = single_frequency_arg_targets + beat_targets + difficult_targets
    return production_targets

#### #### #### #### TEST SYNTHETIC TARGETS #### #### #### #### 

def build_test_targets(duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None) -> List[meta_target.MetaTarget]:
    """builds a list of targets for testing the pipeline"""
    assert None in [sampling_rate, samples], "either sampling rate or number of samples should be specified, but not both"
    
    single_frequency_arg_targets_test = []
    for Target in [meta_target.SineTarget, meta_target.TriangleTarget, meta_target.SawtoothTarget, meta_target.InverseSawtoothTarget, meta_target.SquareTarget]:
        single_frequency_arg_targets_test.append(Target(duration, sampling_rate, samples=samples, freq=100))

    test_beat_target = [meta_target.BeatTarget(duration, sampling_rate, samples=samples, base_freq=100)]

    difficult_targets = [
        meta_target.ChirpTarget(duration, sampling_rate, samples=samples, start_freq=1, stop_freq=1e4),
        meta_target.DampChirpTarget(duration, sampling_rate, samples=samples, start_freq=1, stop_freq=1e4),
        meta_target.SmoothGaussianNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.SmoothUniformNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.GaussianNoiseTarget(duration, sampling_rate, samples=samples),
        meta_target.UniformNoiseTarget(duration, sampling_rate, samples=samples),
    ]

    test_targets = single_frequency_arg_targets_test + test_beat_target + difficult_targets
    return test_targets