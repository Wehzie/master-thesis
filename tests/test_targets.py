import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import meta_target
import data_analysis
import params_target

def test_synthetic_targets(show: bool = False):
    targets = [
        meta_target.SineTarget(1),
        meta_target.TriangleTarget(1),
        meta_target.SawtoothTarget(1),
        meta_target.InverseSawtoothTarget(1),
        meta_target.SquareTarget(1),
        meta_target.ChirpTarget(1, start_freq=1, stop_freq=20, sampling_rate=1000),
        meta_target.BeatTarget(1, base_freq=10),
        meta_target.DampChirpTarget(1, start_freq=1, stop_freq=20, sampling_rate=1000),
        meta_target.SmoothGaussianNoiseTarget(1, sampling_rate=1000),
        meta_target.SmoothUniformNoiseTarget(1, sampling_rate=1000),
        meta_target.GaussianNoiseTarget(1, sampling_rate=1000),
        meta_target.UniformNoiseTarget(1, sampling_rate=1000),
    ]

    if show:
        data_analysis.plot_multiple_targets(targets, show=True)
        data_analysis.plot_multiple_targets_common_axis(targets, show=True)

def test_synthetic_targets_from_parameters(show: bool = False):
    targets = params_target.targets
    if show:
        data_analysis.plot_multiple_targets(targets, show=True)
        data_analysis.plot_multiple_targets_common_axis(targets, show=True)


test_synthetic_targets()