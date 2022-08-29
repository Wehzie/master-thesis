from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Final

from data_analysis import compute_rmse, plot_fourier, plot_signal
from data_io import find_dir_name, json_to_df, load_data, load_sim_data
from data_preprocessor import clean_signal
from params import bird_params
from netlist_generator import build_sum_netlist, run_ngspice, select_netlist_generator

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

class SignalGenerator(ABC):
    
    @staticmethod
    @abstractmethod
    def gen_atomic_signal() -> np.ndarray:
        """generate a signal within a given environment
        
        in Python this would be typically be single oscillator
        in SPICE this would be the a single oscillator or a sum of oscillators"""
        NotImplemented

    @staticmethod
    @abstractmethod
    def sum_atomic_signals() -> np.ndarray:
        """simple summation over atomic signals"""
        NotImplemented