import csv
import numpy as np
from pathlib import Path
from typing import List, Final
from param_types import PythonSignalDetArgs
from sklearn.linear_model import LinearRegression

class Sample():
    """a sample corresponds to a randomly drawn signal approximating the target"""
    def __init__(self, sum_y: np.ndarray,
        signal_matrix: np.ndarray, det_param_li: List[PythonSignalDetArgs]):

                                                    # store single oscillator signals for linear regression
        self.matrix_y: np.ndarray = signal_matrix   #   matrix of single-oscillator signals
        self.sum_y: Final = sum_y   # generated signal y coords
        self.fit_y = None           # summed signal fit to target with regression
        
        # list of determined parameters
        #   one set of parameters corresponds to a single oscillator
        self.det_param_li = det_param_li
        
        self.rmse_sum = None        # root mean square error of summed signal
        self.rmse_fit = None        # root mean square error of regression fit signal
        self.rmse_norm = None       # rmse after normalization of target and sum_y

    def __str__(self) -> str:
        y = f"y:\n{self.sum_y}\n"
        det_param_li = "det_param_li:\n"
        for p in self.det_param_li:
            det_param_li += str(p) + "\n"
        rmse_sum = f"rmse sum: {self.rmse_sum}\n"
        rmse_fit = f"rmse fit: {self.rmse_fit}"
        return y + det_param_li + rmse_sum + rmse_fit

    def save(self, path: Path = "data/best_sample.csv") -> None:
        """save the determined parameters of a sample to a CSV"""
        if len(self.det_param_li) < 1: return
        with open(path, "w") as f:
            writer = csv.writer(f)
            # write header
            writer.writerow(self.det_param_li[0].__dict__)
            # write data
            for osc in self.det_param_li:
                writer.writerow(osc.__dict__.values())

    @staticmethod
    def regress_linear(p: np.ndarray, t: np.ndarray, verbose: bool = False):
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
    def predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
        """generate approximation of target, y"""
        fit = np.sum(X.T * coef, axis=1) + intercept
        return fit