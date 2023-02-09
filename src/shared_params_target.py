"""
This module initializes Synthetic target signals with here-defined parameters.
Furthermore, it provides paths for loading real world signals from file.
"""

from pathlib import Path
from enum import Enum

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

single_frequency_arg_targets = []
for Target in [meta_target.SineTarget, meta_target.TriangleTarget, meta_target.SawtoothTarget, meta_target.InverseSawtoothTarget, meta_target.SquareTarget]:
    for freq in [1e1, 1e2, 1e3, 1e4]:
        single_frequency_arg_targets.append(Target(1, freq=freq))

beat_targets = [
    meta_target.BeatTarget(1, base_freq=1e1),
    meta_target.BeatTarget(1, base_freq=1e2),
    meta_target.BeatTarget(1, base_freq=1e3),
    meta_target.BeatTarget(1, base_freq=1e4),
]

other_targets = [
    meta_target.ChirpTarget(1, start_freq=1, stop_freq=1e4),
    meta_target.DampChirpTarget(1, start_freq=1, stop_freq=1e4),
    meta_target.SmoothGaussianNoiseTarget(1, sampling_rate=1000),
    meta_target.SmoothUniformNoiseTarget(1, sampling_rate=1000),
    meta_target.GaussianNoiseTarget(1, sampling_rate=1000),
    meta_target.UniformNoiseTarget(1, sampling_rate=1000),
]

production_targets = single_frequency_arg_targets + beat_targets + other_targets

#### #### #### #### TEST SYNTHETIC TARGETS #### #### #### #### 

single_frequency_arg_targets_test = []
for Target in [meta_target.SineTarget, meta_target.TriangleTarget, meta_target.SawtoothTarget, meta_target.InverseSawtoothTarget, meta_target.SquareTarget]:
    single_frequency_arg_targets_test.append(Target(1, freq=100))

test_beat_target = [meta_target.BeatTarget(1, base_freq=100)]

test_targets = single_frequency_arg_targets_test + test_beat_target + other_targets