import param_types as party
import data_io
import data_preprocessor

from typing import Final, Union
from pathlib import Path

import numpy as np

class MetaTargetSample:
    """load signal including meta data, sample based processing"""

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
        self.duration = len(target_resampled) / self.sampling_rate
        self.time = np.linspace(0, self.duration, int(self.duration * self.sampling_rate))
    
    def __repr__(self) -> str:
        return f"MetaTarget(signal={self.signal}, sampling_rate={self.sampling_rate}, dtype={self.dtype})"

class MetaTargetTime:
    """load signal including meta data, time based processing"""

    def __init__(self, rand_args: party.SpiceSumRandArgs) -> None:
        # loading and manipulating the target signal
        raw_sampling_rate, raw_target, raw_dtype = data_io.load_data()

        # shorten the target to the desired duration
        new_duration = rand_args.time_stop - rand_args.time_start # seconds
        # TODO: comment in when synthetic signals have been refactored
        # assert 1/raw_sampling_rate < new_duration, "The desired duration is less than the sampling rate"
        time = np.arange(0, new_duration, 1/raw_sampling_rate)
        signal = data_preprocessor.change_duration(raw_target, raw_sampling_rate, new_duration)
        # time returns a non-deterministic number of samples, therefore we need to align the signals
        time, signal = data_preprocessor.align_signals_cut(time, signal)

        self.time = time
        self.signal = signal
        self.sampling_rate = raw_sampling_rate
        self.duration = new_duration
        self.dtype = raw_dtype

    def adjust_samples(self, new_samples: int) -> None:
        """update the signal to the new number of samples"""
        sampling_rate_increase = new_samples / len(self.signal)
        new_sampling_rate = int(self.sampling_rate * sampling_rate_increase) 
        
        # downsample_typesafe gives a much cleaner signal than scipy.resample
        if new_samples < len(self.signal):
            new_signal = data_preprocessor.downsample_typesafe(self.signal, new_samples) # downsample
        else:
            # scipy.resample works well for upsampling
            new_signal = data_preprocessor.resample(self.signal, new_samples) # resample
            new_signal = new_signal.astype(self.dtype) # maintain type

        new_time = np.arange(0, self.duration, 1/new_sampling_rate)
        new_time, new_signal = data_preprocessor.align_signals_cut(new_time, new_signal)

        self.sampling_rate = new_sampling_rate
        self.signal = new_signal
        self.time = new_time
    
    def __repr__(self) -> str:
        return f"MetaTarget(signal={self.signal}, sampling_rate={self.sampling_rate}, dtype={self.dtype})"

UnionMetaTarget = Union[MetaTargetSample, MetaTargetTime]