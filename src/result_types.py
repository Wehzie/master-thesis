from dataclasses import dataclass

import param_types as party

@dataclass
class ResultAlgoSweep:
    algo_name: str
    algo_args: party.AlgoArgs
    mean_rmse: float
    std_rmse: float
    mean_z_ops: float
    std_z_ops: float