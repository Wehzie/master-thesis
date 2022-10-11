from dataclasses import dataclass

import numpy as np

@dataclass
class Target():
    signal: np.ndarray
    sampling_rate: int
    raw_dtype: np.dtype