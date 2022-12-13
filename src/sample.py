from __future__ import annotations

from pathlib import Path
import csv
import pickle
import numpy as np
from typing import List, Final, Union

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import param_types as party
import data_analysis
import data_preprocessor
import meta_target
import data_io

class Sample():
    """a sample consists of n-oscillators with constant parameters
    the sum of oscillators approximates a target signal"""
    def __init__(self, signal_matrix: np.ndarray, weights: Union(None, np.ndarray), weighted_sum: np.ndarray,
        offset: Union(None, float), rmse: float, signal_args: List[party.PythonSignalDetArgs]) -> Sample:
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
        if len(self.signal_args) < 1: return
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

    def update(self, target: np.ndarray) -> None: # TODO: marked for removal
        """recompute sum and rmse"""
        if self.weights is None:
            self.weighted_sum = np.sum(self.signal_matrix, axis=1) + self.offset
        else:
            self.weighted_sum = self.compute_weighted_sum(self.signal_matrix, self.weights, self.offset)
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
        norm_sample = Sample(norm_signal_matrix,
                            norm_weights,
                            norm_signal_sum,
                            norm_offset,
                            norm_rmse,
                            norm_signal_args)
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

def evaluate_prediction(best_sample: Sample, m_target: meta_target.UnionMetaTarget,
    z_ops: int, alg_name: str, plot_time: bool = True, plot_freq: bool = False) -> None:
    """evaluate a generated signal (sample) against the target by qualitative (plots) and quantitative (RMSE) means"""
    # normalize target to range 0 1
    target_norm = data_preprocessor.norm1d(m_target.signal)

    # find best sample and save
    print(f"signal_sum mean: {np.mean(best_sample.weighted_sum)}")
    best_sample.save_sample()
    data_io.save_signal_to_wav(best_sample.weighted_sum, m_target.sampling_rate, m_target.dtype, Path("data/best_sample.wav"))

    norm_sample = Sample.norm_sample(best_sample, target_norm)

    # compute regression against target
    reg_sample = Sample.regress_sample(best_sample, m_target.signal)
    data_io.save_signal_to_wav(reg_sample.weighted_sum, m_target.sampling_rate, m_target.dtype, Path("data/fit.wav"))

    # norm regression after fit (good enough)
    norm_reg_sample = Sample.norm_sample(reg_sample, target_norm)

    # plots
    if plot_time: # time-domain
        #hist_rmse(rmse_list, title="sum distribution")
        #hist_rmse(rmse_norm_list, title="norm-sum distribution")
        data_analysis.plot_pred_target(best_sample.weighted_sum, m_target.signal, time=m_target.time, title=f"{alg_name}, sum")
        data_analysis.plot_pred_target(reg_sample.weighted_sum, m_target.signal, time=m_target.time, title=f"{alg_name}, regression")
        data_analysis.plot_pred_target(norm_sample.weighted_sum, target_norm, time=m_target.time, title=f"{alg_name}, norm-sum")
        data_analysis.plot_pred_target(norm_reg_sample.weighted_sum, target_norm, time=m_target.time, title=f"{alg_name}, norm after fit")
    if plot_freq: # frequency-domain
        data_analysis.plot_fourier(m_target.signal, title=f"{alg_name}, target")
        data_analysis.plot_fourier(best_sample.weighted_sum, title=f"{alg_name}, sum")
        data_analysis.plot_fourier(reg_sample.weighted_sum, title=f"{alg_name}, regression")

    print(f"{alg_name}")
    print(f"z_ops: {z_ops}")
    print(f"best_sample.rmse_sum {best_sample.rmse}")
    print(f"best_sample.rmse_sum-norm {norm_sample.rmse}")
    print(f"best_sample.rmse_fit {reg_sample.rmse}")
    print(f"best_sample.rmse_fit-norm {norm_reg_sample.rmse}")

    plt.show()