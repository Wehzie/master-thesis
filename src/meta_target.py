import param_types as party
import data_io
import data_preprocessor

from typing import Final
from dataclasses import dataclass
from pathlib import Path

import numpy as np

class MetaTarget:
    """signal including meta data"""

    def __init__(self, rand_args: party.PythonSignalRandArgs) -> None:
        # loading and manipulating the target signal
        raw_sampling_rate, raw_target, raw_dtype = data_io.load_data()
        # length checks
        if rand_args.samples > len(raw_target):
            raise ValueError("The desired number of samples is less than contained in the original data")
        # shorten the target
        target_middle = raw_target
        if len(raw_target) // 3 > rand_args.samples:
            target_middle = data_preprocessor.take_middle_third(raw_target)
        # downsample
        target_resampled: Final = data_preprocessor.downsample_typesafe(target_middle, rand_args.samples)
        # save to wav
        data_io.save_signal_to_wav(target_middle, raw_sampling_rate, raw_dtype, Path("data/target_downsampled.wav"))
        # set state
        self.signal = target_resampled
        self.sampling_rate = data_preprocessor.get_sampling_rate_after_resample(target_middle, target_resampled, raw_sampling_rate)
        self.dtype = raw_dtype
    
    def __repr__(self) -> str:
        return f"MetaTarget(signal={self.signal}, sampling_rate={self.sampling_rate}, dtype={self.dtype})"
