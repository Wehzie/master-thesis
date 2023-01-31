"""parameters for generating synthetic target signals and for loading real world signals from file"""

from pathlib import Path
from enum import Enum

import meta_target
import sweep_types as sweety
import params_python
import params_python_test
import params_hybrid
import gen_signal_python
import gen_signal_spipy

#### #### #### #### AUDIO FILE PATHS #### #### #### #### 

class DevSet(Enum): # used for development of the pipeline
    """development set of targets that can be loaded from file"""
    MAGPIE = Path("resources/magpie. 35k, mono, 8-bit, 11025 Hz, 3.3 seconds.wav")
    YES = Path("resources/yes-5.wav")

class TestSet(Enum): # held out until the end of development
    """test set of targets that can be loaded from file"""
    BELLBIRD = Path("resources/white_bellbird.wav")
    OKAY = Path("resources/okay-7.wav")

#### #### #### #### PRODUCTION PARAMETERS #### #### #### #### 

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

python_target_sweep = sweety.TargetSweep(
    "evaluate the ability of the python signal generator to fit a variety of targets",
    production_targets,
    params_python.py_rand_args_uniform,
    gen_signal_python.PythonSigGen(),
    max_z_ops=1e4,
    m_averages=10
)
hybrid_target_sweep = sweety.TargetSweep(
    "evaluate the ability of the hybrid signal generator to fit a variety of targets",
    production_targets,
    params_hybrid.spice_rand_args_uniform,
    gen_signal_spipy.SpipySignalGenerator(),
    max_z_ops=1e4,
    m_averages=10
)

#### #### #### #### TEST PARAMETERS #### #### #### #### 

single_frequency_arg_targets_test = []
for Target in [meta_target.SineTarget, meta_target.TriangleTarget, meta_target.SawtoothTarget, meta_target.InverseSawtoothTarget, meta_target.SquareTarget]:
    single_frequency_arg_targets_test.append(Target(1, freq=100))

test_beat_target = [meta_target.BeatTarget(1, base_freq=100)]

test_targets = single_frequency_arg_targets_test + test_beat_target + other_targets

python_target_sweep_test = sweety.TargetSweep(
    "a reduced set of targets for testing the python signal generator against variety of targets",
    test_targets,
    params_python_test.py_rand_args_uniform,
    gen_signal_python.PythonSigGen(),
    max_z_ops=300,
    m_averages=2
)

hybrid_target_sweep_test = sweety.TargetSweep(
    "a reduced set of targets for testing the hybrid signal generator against variety of targets",
    test_targets,
    params_hybrid.spice_rand_args_uniform,
    gen_signal_spipy.SpipySignalGenerator(),
    max_z_ops=300,
    m_averages=2
)

#### #### #### #### SAMPLE BASED PYTHON-GENERATOR TEST PARAMETERS #### #### #### #### 

sample_targets = [
    meta_target.SineTarget(0.1, freq=1, samples=30),
    meta_target.TriangleTarget(0.1, freq=1, samples=30),
    # meta_target.SawtoothTarget(0.1, freq=1, samples=30),
    # meta_target.InverseSawtoothTarget(0.1, freq=1, samples=30),
    # meta_target.SquareTarget(0.1, freq=1, samples=30),
    # meta_target.BeatTarget(0.1, base_freq=1, samples=30),
    # meta_target.ChirpTarget(0.1, start_freq=1, stop_freq=10, samples=30),
    # meta_target.DampChirpTarget(0.1, start_freq=1, stop_freq=10, samples=30),
    # meta_target.SmoothGaussianNoiseTarget(0.1, samples=30),
    # meta_target.SmoothUniformNoiseTarget(0.1, samples=30),
    # meta_target.GaussianNoiseTarget(0.1, samples=30),
    # meta_target.UniformNoiseTarget(0.1, samples=30),
]

python_target_sweep_sample = sweety.TargetSweep(
    "sample based targets before making the python signal generator adapt its sampling rate to the target",
    sample_targets,
    params_python_test.py_rand_args_uniform,
    gen_signal_python.PythonSigGen(),
    max_z_ops=300,
    m_averages=2
)