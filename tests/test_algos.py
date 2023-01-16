
# add code in src to path
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from main import *

def test_lazy():
    rand_args, meta_target = params_python.init_target2rand_args()
    target = meta_target[1]
    algo_sweep = params_python.init_algo_sweep(target)
    simple_algo_sweep(algo_sweep, *meta_target)