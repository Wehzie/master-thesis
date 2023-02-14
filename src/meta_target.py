"""
This module implements the abstract MetaTarget class and subclasses.
The MetaTarget class bundles a signal with its time axis and other metadata.
The subclasses implement different ways to load a signal from file or generate a signal.
"""

import gen_signal_args_types as party
import data_io
import data_preprocessor
import const

from typing import Final, Union
from pathlib import Path
from abc import ABC

import numpy as np
from scipy import signal
from scipy import interpolate

#### #### #### #### REAL WORLD TARGETS #### #### #### ####

class MetaTarget(ABC):
    """abstract base class for all meta targets"""

    def __init__(self, signal: np.ndarray, time: np.ndarray, sampling_rate: int, dtype: np.dtype, duration: float, name: str) -> None:
        self.signal = signal
        self.time = time
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.duration = duration
        self.samples = len(signal)
        self.name = name
    
    def __repr__(self) -> str:
        return f"MetaTarget({self.name}, sampling_rate={self.sampling_rate}, dtype={self.dtype}, duration={self.duration})"
    
    def sinc_interpolate(self, new_sampling_rate: int):
        new_samples = np.round(self.duration * new_sampling_rate).astype(int)
        new_time = np.linspace(0, self.duration, new_samples)

        self.signal = data_preprocessor.interpolate_sinc_time(self.signal, self.time, new_time)
        self.time = new_time
        self.sampling_rate = new_sampling_rate
        self.samples = new_samples

class MetaTargetSample(MetaTarget):
    """load signal from file, save meta data; use sample based processing"""

    def __init__(self, rand_args: party.PythonSignalRandArgs, name: str, path: Path) -> None:
        self.name = name
        # loading and manipulating the target signal
        raw_sampling_rate, raw_target, raw_dtype = data_io.load_data(path)
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
        self.samples = len(self.signal)
        self.sampling_rate = data_preprocessor.get_sampling_rate_after_resample(target_middle, target_resampled, raw_sampling_rate)
        self.dtype = raw_dtype
        self.duration = len(target_resampled) / self.sampling_rate
        self.time = np.linspace(0, self.duration, np.around(self.duration * self.sampling_rate).astype(int), endpoint=False)
    
    def __repr__(self) -> str:
        return f"MetaTargetSample(samples={len(self.signal)}, duration={self.duration}, sampling_rate={self.sampling_rate}, dtype={self.dtype})"

class MetaTargetTime(MetaTarget):
    """load signal from file, save meta data; use time based processing"""

    def __init__(self, rand_args: party.SpiceSumRandArgs, name: str, path: Path) -> None:
        self.name = name
        # loading and manipulating the target signal
        raw_sampling_rate, raw_target, raw_dtype = data_io.load_data(path)

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
        self.samples = len(signal)

        if len(signal) < 1:
            raise ValueError("The selected target duration is less than the distance between two samples")

        # resample the time-aligned signal to the number of samples generated by spice 
        spice_samples = rand_args.estimate_number_of_samples()
        self.adjust_samples(spice_samples)

    def adjust_samples(self, new_samples: int) -> None:
        """update the signal to the new number of samples"""
        print(f"adjusting samples from {len(self.signal)} to {new_samples}")
        sampling_rate_increase = new_samples / len(self.signal)
        new_sampling_rate = np.around(self.sampling_rate * sampling_rate_increase).astype(int)
        
        # downsample_typesafe gives a much cleaner signal than scipy.resample
        if new_samples < len(self.signal):
            new_signal = data_preprocessor.downsample_typesafe(self.signal, new_samples) # downsample
        else:
            # scipy.resample works well for upsampling
            new_signal = data_preprocessor.resample(self.signal, new_samples) # resample
            new_signal = new_signal.astype(self.dtype) # maintain type

        new_time = np.arange(0, self.duration, 1/new_sampling_rate)
        new_time, new_signal = data_preprocessor.align_signals_cut(new_time, new_signal)

        self.time = new_time
        self.signal = new_signal
        self.sampling_rate = new_sampling_rate
        self.samples = len(new_signal)

    # TODO: should look smoother than the result of scipy.resample with adjust_samples
    def upsample_by_interpolation(signal: np.ndarray, sampling_rate: int, new_sampling_rate: int) -> np.ndarray:
        """upsample a signal by interpolation"""
        time = np.arange(0, len(signal)/sampling_rate, 1/sampling_rate)
        new_time = np.arange(0, len(signal)/sampling_rate, 1/new_sampling_rate)
        new_signal = interpolate.interp1d(time, signal, kind='cubic')(new_time)
        return new_signal, new_time
    
    def __repr__(self) -> str:
        return f"MetaTarget(signal={self.signal}, sampling_rate={self.sampling_rate}, dtype={self.dtype})"



#### #### #### #### SYNTHETIC TARGETS #### #### #### ####

class SyntheticTarget(MetaTarget):
    
    def __init__(self, duration: float,
    sampling_rate: Union[int, None] = None,
    samples: Union[int, None] = None,
    max_freq: Union[float, None] = None) -> None:
        """abstract class for synthetic targets

        args:
            duration: the duration of the signal in seconds
            sampling_rate: the sampling rate of the signal
            samples: the number of samples in the signal
            max_freq: the maximum frequency in the signal
        """
        self.dtype = np.float64
        self.duration = duration
        self.derive_samples_or_sampling_rate(duration, samples, sampling_rate, max_freq)
        self.time = np.linspace(0, duration, self.samples, endpoint=False)

    def compute_oversampling_rate(self, max_freq: float) -> int:
        """compute the oversampling rate for the given sampling rate
        
        args:
            max_freq: the maximum frequency in the signal
        """
        nyquist_rate = max_freq * 2
        return np.around(nyquist_rate * const.OVERSAMPLING_FACTOR).astype(int)

    def compute_nyquist_rate(self, max_freq: float) -> int:
        """compute the nyquist rate for a given signal
        
        args:
            max_freq: the maximum frequency in the signal
        """
        nyquist_rate = max_freq * 2
        return nyquist_rate
        

    def derive_samples_or_sampling_rate(self, duration: float, samples: int, sampling_rate: int, max_freq: float) -> None:
        """given a duration infer the number of samples samples or sampling rate"""
        print(self.__class__.__name__)
        print(duration, samples, sampling_rate)
        if duration and sampling_rate and max_freq:
            assert sampling_rate >= self.compute_nyquist_rate(max_freq), "sampling rate is too low for the given max frequency"
            self.sampling_rate = sampling_rate
            self.samples = np.around(self.sampling_rate * duration).astype(int)
            return

        if duration and sampling_rate:
            self.sampling_rate = sampling_rate
            self.samples = np.around(self.sampling_rate * duration).astype(int)
            return
        
        if samples and max_freq:
            self.samples = samples
            self.sampling_rate = np.around(samples*1/duration).astype(int)
            return

        assert len(set([samples, sampling_rate, max_freq])) == 2, "only one of samples, sampling rate or max_freq should be specified at this point"
        if max_freq:
            self.sampling_rate = self.compute_nyquist_rate(max_freq)
            self.samples = np.around(self.sampling_rate * duration).astype(int)
        elif samples:
            self.samples = samples
            self.sampling_rate = np.around(self.samples * 1/duration).astype(int)
        elif sampling_rate: # sampling rate is not None
            self.sampling_rate = sampling_rate
            self.samples = np.around(self.sampling_rate * duration).astype(int)
        else:
            raise ValueError("checks have failed: only one of samples, sampling rate or max_freq should be specified")

    def moving_average(self, arr: np.ndarray, window_length: int) -> np.ndarray:
        """compute the l-point average over the signal using a convolution"""
        unpadded = np.convolve(arr, np.ones(window_length), "valid") / window_length
        return data_preprocessor.pad_zero(unpadded, len(arr))


class SineTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None, freq: float = 1, amplitude: float = 1, phase: float = 0, offset: float = 0) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "sine"
        self.signal = np.sin(2 * np.pi * freq * self.time + phase*np.pi) * amplitude + offset

class TriangleTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None, freq: float = 1, amplitude: float = 1, phase: float = 0, offset: float = 0) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "triangle"
        self.signal = signal.sawtooth(2 * np.pi * freq * self.time + phase*np.pi, width=0.5) * amplitude + offset

class SquareTarget(SyntheticTarget):
    
    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None, freq: float = 1, amplitude: float = 1, phase: float = 0, offset: float = 0) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "square"
        self.signal = np.sign(np.sin(2 * np.pi * freq * self.time + phase*np.pi)) * amplitude + offset

class SawtoothTarget(SyntheticTarget):
    
    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None, freq: float = 1, amplitude: float = 1, phase: float = 0, offset: float = 0) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "sawtooth"
        self.signal = signal.sawtooth(2 * np.pi * freq * self.time + phase*np.pi) * amplitude + offset

class InverseSawtoothTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None, freq: float = 1, amplitude: float = 1, phase: float = 0, offset: float = 0) -> None:
        super().__init__(duration, sampling_rate, samples, freq)
        self.name = "inverse_sawtooth"
        self.signal = -signal.sawtooth(2 * np.pi * freq * self.time + phase*np.pi) * amplitude + offset

class ChirpTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None, start_freq: float = 0.1, stop_freq: Union[float, None] = None, amplitude: float = 1, offset: float = 0) -> None:
        """
        initialize a chirp signal.

        args:
            duration: the duration of the signal in seconds
            sampling_rate: the sampling rate of the signal,
                           when None the sampling rate is derived from the maximum frequency in the signal
            samples: the number of samples in the signal,
                     when None the number of samples is derived from the sampling rate and duration
            start_freq: the start frequency of the chirp
            stop_freq: the stop frequency of the chirp,
                       when None the stop frequency is derived from the sampling rate   
        """
        super().__init__(duration, sampling_rate, samples, stop_freq)
        self.name = "chirp"
        if stop_freq is None:
            stop_freq = self.sampling_rate/20
        self.signal = signal.chirp(self.time, start_freq, self.duration, stop_freq) * amplitude + offset

class BeatTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None,
    base_freq: float = 1, base_freq_factor: float = 1/10,
    amplitude: float = 1, phase: float = 0, offset: float = 0) -> None:
        """
        generate a beat note signal that is the product of two sinusoids with different frequencies
        
        args:
            base_freq: the first of two frequency components of the beat note
            base_freq_factor: the frequency of the second frequency components is the product of the base_frequency and the base_freq_factor
        """
        super().__init__(duration, sampling_rate, samples, base_freq)
        self.name = "beat"
        derived_freq = base_freq * base_freq_factor
        self.signal = np.sin(2 * np.pi * base_freq * self.time + phase*np.pi) * np.sin(2 * np.pi * derived_freq * self.time + phase*np.pi) * amplitude + offset

class DampChirpTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None, start_freq: float = 0.1, stop_freq: Union[float, None] = None, amplitude: float = 1, offset: float = 0) -> None:
        """
        initialize a chirp signal.

        args:
            duration: the duration of the signal in seconds
            sampling_rate: the sampling rate of the signal,
                           when None the sampling rate is derived from the maximum frequency in the signal
            samples: the number of samples in the signal,
                     when None the number of samples is derived from the sampling rate and duration
            start_freq: the start frequency of the chirp
            stop_freq: the stop frequency of the chirp,
                       when None the stop frequency is derived from the sampling rate   
        """
        super().__init__(duration, sampling_rate, samples, stop_freq)
        self.name = "damp_chirp"
        if stop_freq is None:
            stop_freq = self.sampling_rate/20
        self.signal = signal.chirp(self.time, start_freq, self.duration, stop_freq) * amplitude + offset
        self.signal = self.signal * np.exp(-self.time)

class SmoothGaussianNoiseTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None,
    amplitude: float = 1, offset: float = 0, avg_window: int = 10) -> None:
        super().__init__(duration, sampling_rate, samples)
        self.name = "smooth_gaussian_noise"
        self.signal = const.RNG.normal(0, 1, self.samples) * amplitude + offset
        self.signal = self.moving_average(self.signal, avg_window)

class SmoothUniformNoiseTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None,
    amplitude: float = 1, offset: float = 0, avg_window: int = 10) -> None:
        super().__init__(duration, sampling_rate, samples)
        self.name = "smooth_uniform_noise"
        self.signal = const.RNG.uniform(-1, 1, self.samples) * amplitude + offset
        self.signal = self.moving_average(self.signal, avg_window)

class GaussianNoiseTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None,
    amplitude: float = 1, offset: float = 0) -> None:
        super().__init__(duration, sampling_rate, samples)
        self.name = "gaussian_noise"
        self.signal = const.RNG.normal(0, 1, self.samples) * amplitude + offset

class UniformNoiseTarget(SyntheticTarget):

    def __init__(self, duration: float, sampling_rate: Union[int, None] = None, samples: Union[int, None] = None,
    amplitude: float = 1, offset: float = 0) -> None:
        super().__init__(duration, sampling_rate, samples)
        self.name = "uniform_noise"
        self.signal = const.RNG.uniform(-1, 1, self.samples) * amplitude + offset
