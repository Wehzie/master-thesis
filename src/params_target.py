"""parameters for generating synthetic target signals and for loading real world signals from file"""

import meta_target

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

targets = single_frequency_arg_targets + beat_targets + other_targets