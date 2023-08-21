"""
This module implements the sample class which is equivalent to an oscillator ensemble.

The sum of n-oscillators approximates a target signal.
A sample is tuned by an algorithm to minimize the rmse between the sum and the target.
"""

from __future__ import annotations
import copy

from pathlib import Path
import csv
import pickle
import numpy as np
from typing import List, Union

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import gen_signal_args_types as party
import data_analysis
import data_preprocessor
import meta_target
import data_io
import const


class Sample:
    """
    A sample is an oscillator ensemble.

    The term sample is used to indicate that we draw from a random variable.
    Thus, a sample consists of n-oscillators with constant parameters.
    The sum of oscillators approximates a target signal.
    """

    def __init__(
        self,
        signal_matrix: np.ndarray,
        weights: Union[None, np.ndarray],
        weighted_sum: np.ndarray,
        offset: Union[None, float],
        rmse: float,
        signal_args: Union[List[None], List[party.PythonSignalDetArgs]],
    ) -> Sample:
        """
        initialize a sample

        param:
            signal_matrix:  matrix of single-oscillator signals
            weights:        array of weights over signal_matrix
            weighted_sum:     apply weights to signal_matrix then sum and add offset
            offset:         offset over matrix
            rmse:           rmse(signal_sum, target)
            signal_args:    list of parameters generating the signal matrix
                                one set of parameters corresponds to a single oscillator

        return:
            Sample
        """
        self.signal_matrix = signal_matrix
        self.weights = weights
        self.weighted_sum = weighted_sum
        self.offset = offset
        self.rmse = rmse
        self.signal_args = signal_args
        # time axis is stored in the target or signal generation args (rand_args)

    def __str__(self) -> str:
        signal_matrix = f"signal_matrix:\n{self.signal_matrix}\n"
        weights = f"weights:\n{self.weights}\n"
        weighted_sum = f"weighted_sum:\n{self.weighted_sum}\n"
        offset = f"offset:\n{self.offset}\n"
        rmse = f"rmse: {self.rmse}\n"
        signal_args = "signal_args:\n"
        for args in self.signal_args:
            signal_args += str(args) + "\n"
        return signal_matrix + signal_args + weighted_sum + offset + weights + rmse

    def save_signal_args(self, path: Path = "data/best_sample.csv") -> None:
        """save the determined parameters of a sample to a CSV"""
        if len(self.signal_args) < 1:
            return
        with open(path, "w") as f:
            writer = csv.writer(f)
            # write header
            writer.writerow(self.signal_args[0].__dict__)
            # write data
            for osc in self.signal_args:
                writer.writerow(osc.__dict__.values())

    def save_sample(self, data_path: Path = "data/best_sample.pickle") -> None:
        """serialize a sample and save to file"""
        with open(data_path, "wb") as f:
            pickle.dump(self, f)

    def update(self, target: np.ndarray) -> None:  # TODO: marked for removal
        """recompute sum and rmse"""
        if self.weights is None:
            self.weighted_sum = np.sum(self.signal_matrix, axis=1) + self.offset
        else:
            self.weighted_sum = self.compute_weighted_sum(
                self.signal_matrix, self.weights, self.offset
            )
        self.rmse = data_analysis.compute_rmse(self.weighted_sum, target)

    @staticmethod
    def regress1d(p: np.ndarray, t: np.ndarray, verbose: bool = False):
        """apply linear regression"""
        # matrix_y refers to the y-values of a generated signal
        # the individual signals are used as regressors

        r = p.T

        if verbose:
            print("Computing regression")
            print(f"r {r}")
            print(f"r.shape {r.shape}")
            print(f"t {t}")
            print(f"t.shape {t.shape}")

        reg = LinearRegression().fit(r, t)

        if verbose:
            print(f"Coefficient: {reg.coef_}")
            print(f"Coefficient.shape: {reg.coef_.shape}")
            print(f"Intercept: {reg.intercept_}")

        return reg

    @staticmethod
    def compute_weighted_sum(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
        """generate approximation of target, y"""
        fit = np.sum(X.T * coef, axis=1) + intercept
        return fit

    @staticmethod
    def norm_sample(sample: Sample, target: np.ndarray) -> Sample:
        """normalize the fields of a sample"""
        norm_signal_matrix = data_preprocessor.norm2d(sample.signal_matrix)
        if sample.weights is not None:
            norm_weights = data_preprocessor.norm1d(sample.weights)
        else:
            norm_weights = None
        norm_signal_sum = data_preprocessor.norm1d(sample.weighted_sum)
        norm_offset = 0
        norm_rmse = data_analysis.compute_rmse(norm_signal_sum, target)
        norm_signal_args = sample.signal_args
        norm_sample = Sample(
            norm_signal_matrix,
            norm_weights,
            norm_signal_sum,
            norm_offset,
            norm_rmse,
            norm_signal_args,
        )
        return norm_sample

    @staticmethod
    def regress_sample(sample: Sample, target: np.ndarray) -> Sample:
        """apply linear regression to a sample to fit against target"""
        reg = Sample.regress1d(sample.signal_matrix, target)
        signal_matrix = sample.signal_matrix
        weights = reg.coef_
        offset = reg.intercept_
        weighted_sum = Sample.compute_weighted_sum(signal_matrix, weights, offset)
        rmse = data_analysis.compute_rmse(weighted_sum, target)
        return Sample(signal_matrix, weights, weighted_sum, offset, rmse, sample.signal_args)


def evaluate_prediction(
    best_sample: Sample,
    m_target: meta_target.MetaTarget,
    z_ops: int,
    alg_name: str,
    generator_name: str,
    plot_time: bool = True,
    plot_freq: bool = True,
    decompose_sample: bool = True,
    interpolate: bool = True,
    write_dir: Path = const.WRITE_DIR,
) -> None:
    """evaluate a generated signal (sample) against the target by qualitative (plots) and quantitative (RMSE) means"""
    m_target = copy.deepcopy(
        m_target
    )  # local copy to avoid side effects when running multiple times
    save_path = data_io.find_dir_name(
        write_dir, f"qualitative_{generator_name}_{m_target.__class__.__name__}_{alg_name}"
    )

    n_osc = best_sample.signal_matrix.shape[0]

    # normalize target to range 0 1
    target_norm = data_preprocessor.norm1d(m_target.signal)

    # find best sample and save
    print(f"signal_sum mean: {np.mean(best_sample.weighted_sum)}")
    best_sample.save_sample()
    data_io.save_signal_to_wav(
        best_sample.weighted_sum,
        m_target.sampling_rate,
        m_target.dtype,
        Path("data/best_sample.wav"),
    )

    norm_sample = Sample.norm_sample(best_sample, target_norm)

    # compute regression against target
    reg_sample = Sample.regress_sample(best_sample, m_target.signal)
    data_io.save_signal_to_wav(
        reg_sample.weighted_sum, m_target.sampling_rate, m_target.dtype, Path("data/fit.wav")
    )

    # norm regression after fit (good enough)
    norm_reg_sample = Sample.norm_sample(reg_sample, target_norm)

    # plots
    if plot_time:  # time-domain
        time_dir = save_path / "time_domain"
        time_dir.mkdir(parents=True, exist_ok=True)
        data_analysis.plot_signal(
            m_target.signal,
            m_target.time,
            title=f"{alg_name}, n={n_osc}, z={z_ops}",
            save_path=time_dir / "target_alone",
        )
        data_analysis.plot_signal(
            best_sample.weighted_sum,
            m_target.time,
            title=f"{alg_name}, n={n_osc}, z={z_ops}",
            save_path=time_dir / "base_algorithm_without_target",
        )
        data_analysis.plot_pred_target(
            best_sample.weighted_sum,
            m_target.signal,
            time=m_target.time,
            title=f"{alg_name}, n={n_osc}, z={z_ops}",
            save_path=time_dir / "base_algorithm",
        )
        data_analysis.plot_pred_target(
            reg_sample.weighted_sum,
            m_target.signal,
            time=m_target.time,
            title=f"regression after {alg_name}, n={n_osc}, z={z_ops}",
            save_path=time_dir / "regression",
        )
        data_analysis.plot_pred_target(
            norm_sample.weighted_sum,
            target_norm,
            time=m_target.time,
            title=f"{alg_name}, normalized, n={n_osc}, z={z_ops}",
            save_path=time_dir / "normalized_base_algorithm",
        )
        data_analysis.plot_pred_target(
            norm_reg_sample.weighted_sum,
            target_norm,
            time=m_target.time,
            title=f"normalized after regression, n={n_osc}, z={z_ops}",
            save_path=time_dir / "normalized_regression",
        )
        plt.close("all")
    if plot_freq:  # frequency-domain
        freq_dir = save_path / "frequency_domain"
        freq_dir.mkdir(parents=True, exist_ok=True)
        data_analysis.plot_fourier(
            m_target.signal,
            title=f"{alg_name}, target, n={n_osc}, z={z_ops}",
            save_path=freq_dir / "target",
        )
        data_analysis.plot_fourier(
            best_sample.weighted_sum,
            title=f"{alg_name}, sum, n={n_osc}, z={z_ops}",
            save_path=freq_dir / "sum",
        )
        data_analysis.plot_fourier(
            reg_sample.weighted_sum,
            title=f"{alg_name}, regression, n={n_osc}, z={z_ops}",
            save_path=freq_dir / "regression",
        )
        plt.close("all")
    if decompose_sample:  # show individual signals in best sample
        data_analysis.plot_individual_oscillators(
            best_sample.signal_matrix, m_target.time, save_path=time_dir / "individual_signals"
        )
        data_analysis.plot_f0_hist(
            best_sample.signal_matrix,
            1 / m_target.sampling_rate,
            title=f"fundamental frequency distribution, n={n_osc}, z={z_ops}",
            save_path=time_dir / "frequency_distribution",
        )
        data_analysis.plot_weight_hist(
            best_sample.weights,
            title=f"weight distribution, n={n_osc}, z={z_ops}",
            save_path=time_dir / "weight_distribution",
        )
        plt.close("all")

    max_freq = ""
    if isinstance(m_target, meta_target.ChirpTarget) or isinstance(
        m_target, meta_target.DampChirpTarget
    ):
        max_freq = (
            f"target start freq: {m_target.start_freq}\ntarget stop freq: {m_target.stop_freq}"
        )
    elif isinstance(m_target, meta_target.SyntheticTarget):
        max_freq = f"target max freq: {m_target.max_freq}"

    out = f"""
{alg_name}
n_osc: {n_osc}
z_ops: {z_ops}
duration: {m_target.duration}
{alg_name} RMSE: {best_sample.rmse}
{alg_name} normalized RMSE: {norm_sample.rmse}
regression after {alg_name} RMSE: {reg_sample.rmse}
regression after {alg_name} normalized RMSE: {norm_reg_sample.rmse}
target name: {m_target.name}
target duration: {m_target.duration}
target sampling rate: {m_target.sampling_rate}
{max_freq}
    """
    data_io.save_object_to_string(out, save_path / "results.txt")
    print(out)

    if interpolate:  # apply sinc interpolation on time domain signals
        new_sampling_rate = np.round(m_target.sampling_rate * const.OVERSAMPLING_FACTOR).astype(int)
        interpol_sum = data_preprocessor.interpolate_sinc_sampling_rate(
            best_sample.weighted_sum, m_target.sampling_rate, new_sampling_rate
        )
        interpol_reg = data_preprocessor.interpolate_sinc_sampling_rate(
            reg_sample.weighted_sum, m_target.sampling_rate, new_sampling_rate
        )
        m_target.signal = data_preprocessor.interpolate_sinc_sampling_rate(
            m_target.signal, m_target.sampling_rate, new_sampling_rate
        )

        new_samples = np.round(m_target.duration * new_sampling_rate).astype(int)
        new_time = np.linspace(0, m_target.duration, new_samples, endpoint=False)
        if len(new_time) > len(m_target.signal):
            new_time = new_time[0 : len(m_target.signal)]

        data_analysis.plot_pred_target(
            interpol_sum,
            m_target.signal,
            time=new_time,
            title=f"{alg_name}, interpolated, n={n_osc}, z={z_ops}",
            save_path=time_dir / "interpolated_sum",
        )
        data_analysis.plot_pred_target(
            interpol_reg,
            m_target.signal,
            time=new_time,
            title=f"regression after {alg_name}, interpolated, n={n_osc}, z={z_ops}",
            save_path=time_dir / "interpolated_regression",
        )

    plt.close("all")
