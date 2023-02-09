"""
The main module serves as the entry point to load parameters and start a simulation.
"""

import const
import meta_target
import data_analysis
import experimenteur
if const.TEST_PARAMS:
    import params_python_test as params_python
    import params_hybrid_test as params_hybrid
else:
    import params_python
    import params_hybrid
import params_hybrid
import shared_params_target
import gen_signal_python
import gen_signal_spipy
import sweep_builder


@data_analysis.print_time
def main():
    # Python
    exp = experimenteur.Experimenteur()
    if True:
        sig_gen = gen_signal_python.PythonSigGen()
        generator_args = params_python.py_rand_args_uniform
        m_target = meta_target.MetaTargetSample(generator_args, "magpie", shared_params_target.DevSet.MAGPIE.value)
        sweep_bundle = sweep_builder.bundle_python_sweep(sig_gen, generator_args, m_target, algo_selector="all")

        exp.run_qualitative_algo_sweep(sweep_bundle.algo_sweep, m_target, visual=False)
        exp.run_all_experiments(sweep_bundle, m_target.samples, generator_args)

    # SpiPy
    if True:
        generator_args = params_hybrid.spice_rand_args_uniform
        m_target = meta_target.MetaTargetTime(generator_args, "magpie", shared_params_target.DevSet.MAGPIE.value)

        # scale the number of samples in the target to the number of samples produced by spice
        sig_gen = gen_signal_spipy.SpipySignalGenerator()
        spice_samples = sig_gen.estimate_number_of_samples(generator_args)
        m_target.adjust_samples(spice_samples)

        sweep_bundle = sweep_builder.bundle_python_sweep(sig_gen, generator_args, m_target, algo_selector="all")

        exp.run_qualitative_algo_sweep(sweep_bundle.algo_sweep, m_target, visual=False)
        # exp.run_all_experiments(sweep_bundle, m_target.samples, generator_args)

if __name__ == "__main__":
    main()
