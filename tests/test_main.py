import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import main
import param_util
import meta_target
import params_test_py
import params_test_spipy
import gen_signal_spipy

def test_qualitative_sweep_python():
    rand_args = params_test_py.py_rand_args_uniform
    m_target = meta_target.MetaTargetSample(rand_args)
    algo_sweep_test = param_util.init_algo_sweep(m_target.signal, rand_args, max_z_ops=300, test_mode=True)
    main.qualitative_algo_sweep(algo_sweep_test, m_target, visual=False)

def test_quantitative_sweep_python():
    rand_args = params_test_py.py_rand_args_uniform
    m_target = meta_target.MetaTargetSample(rand_args)
    algo_sweep_test = param_util.init_algo_sweep(m_target.signal, rand_args, max_z_ops=300, m_averages=2, test_mode=True)
    main.produce_all_results(algo_sweep_test, m_target.signal, rand_args)

def test_qualitative_sweep_spipy():
    rand_args = params_test_spipy.spice_rand_args_uniform
    m_target = meta_target.MetaTargetTime(rand_args)
    signal_generator = gen_signal_spipy.SpipySignalGenerator()
    spice_samples = signal_generator.estimate_number_of_samples(rand_args)
    m_target.adjust_samples(spice_samples)
    algo_sweep = param_util.init_algo_sweep(m_target.signal, rand_args, sig_generator=signal_generator, max_z_ops=500)
    main.qualitative_algo_sweep(algo_sweep, m_target, visual=False)