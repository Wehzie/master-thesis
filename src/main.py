import const
import sample
import meta_target
import data_analysis
import data_io
import experimenteur
import experiment_analysis as expan
if const.TEST_PARAMS:
    import params_python_test as params_python
else:
    import params_python
import param_types as party
import params_hybrid
import param_util
import params_target
import sweep_types as sweety
import gen_signal_spipy


def qualitative_algo_sweep(algo_sweep: sweety.AlgoSweep, m_target: meta_target.MetaTarget, visual: bool = False) -> None:
    """algo sweep without averaging or collecting results.
    plots the best sample for each algorithm against the target."""
    for awa in algo_sweep.algo_with_args:
        awa: sweety.AlgoWithArgs
        search_alg = awa.Algo(awa.algo_args)
        best_sample, z_ops = search_alg.search()
        if visual: sample.evaluate_prediction(best_sample, m_target, z_ops, search_alg.__class__.__name__)

@data_analysis.print_time
def produce_all_results(algo_sweep: sweety.AlgoSweep, target_samples: int, base_rand_args: party.PythonSignalRandArgs) -> None:
    """run all experiments and plot results"""
    show_all = False
    exp = experimenteur.Experimenteur()

    exp.set_sweep_name_and_dir("targets_vs_rmse")
    results = exp.run_target_sweep(params_python.python_target_sweep_sample)
    df = expan.conv_results_to_pd(results)
    expan.analyze_targets_vs_rmse(df, exp.sweep_name, exp.sweep_dir, show=show_all)
    data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

    exp.set_sweep_name_and_dir("n_osc_vs_rmse")
    results = exp.run_rand_args_sweep(algo_sweep, params_python.n_osc_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_n_vs_rmse(df, target_samples, exp.sweep_name, exp.sweep_dir, show=show_all)
    expan.plot_masks(algo_sweep.algo_masks, expan.plot_n_vs_rmse, df, target_samples, exp.sweep_name, exp.sweep_dir, show=show_all)
    data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

    exp.set_sweep_name_and_dir("z_ops_vs_rmse")
    results = exp.run_z_ops_sweep(algo_sweep, params_python.z_ops_sweep)
    df = expan.conv_results_to_pd(results)
    expan.plot_z_vs_rmse(df, target_samples, exp.sweep_name, exp.sweep_dir, show=show_all)
    expan.plot_masks(algo_sweep.algo_masks, expan.plot_z_vs_rmse, df, target_samples, exp.sweep_name, exp.sweep_dir, show=show_all)
    data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

    exp.set_sweep_name_and_dir("samples_vs_rmse")
    results = exp.run_sampling_rate_sweep(params_python.sampling_rate_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_samples_vs_rmse(df, exp.sweep_name, exp.sweep_dir, show=show_all)
    expan.plot_masks(algo_sweep.algo_masks, expan.plot_samples_vs_rmse, df, exp.sweep_name, exp.sweep_dir, show=show_all)
    data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

    freq_sweeps = [params_python.freq_sweep_from_zero, params_python.freq_sweep_around_vo2]
    freq_sweep_names = ["freq_range_from_zero", "freq_range_around_vo2"]
    for freq_sweep, freq_sweep_name in zip(freq_sweeps, freq_sweep_names):
        exp.set_sweep_name_and_dir(freq_sweep_name)
        results = exp.run_rand_args_sweep(algo_sweep, freq_sweep, base_rand_args)
        df = expan.conv_results_to_pd(results)
        expan.plot_freq_range_vs_rmse(df, target_samples, freq_sweep_name, exp.sweep_dir, show=show_all)
        expan.plot_masks(algo_sweep.algo_masks, expan.plot_freq_range_vs_rmse, df, target_samples, freq_sweep_name, exp.sweep_dir, show=show_all)
        data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

    exp.set_sweep_name_and_dir("weight_range_vs_rmse")
    results = exp.run_rand_args_sweep(algo_sweep, params_python.weight_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_weight_range_vs_rmse(df, target_samples, exp.sweep_dir, show=show_all)
    expan.plot_masks(algo_sweep.algo_masks, expan.plot_weight_range_vs_rmse, df, target_samples,exp.sweep_dir, show=show_all)
    data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

    exp.set_sweep_name_and_dir("phase_range_vs_rmse")
    results = exp.run_rand_args_sweep(algo_sweep, params_python.phase_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_phase_range_vs_rmse(df, target_samples, exp.sweep_dir, show=show_all)
    expan.plot_masks(algo_sweep.algo_masks, expan.plot_phase_range_vs_rmse, df, target_samples, exp.sweep_dir, show=show_all)
    data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

    exp.set_sweep_name_and_dir("offset_range_vs_rmse")
    results = exp.run_rand_args_sweep(algo_sweep, params_python.offset_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_offset_range_vs_rmse(df, target_samples, exp.sweep_dir, show=show_all)
    expan.plot_masks(algo_sweep.algo_masks, expan.plot_offset_range_vs_rmse, df, target_samples, exp.sweep_dir, show=show_all)
    data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

    exp.set_sweep_name_and_dir("amplitude_vs_rmse")
    results = exp.run_rand_args_sweep(algo_sweep, params_python.amplitude_sweep, base_rand_args)
    df = expan.conv_results_to_pd(results)
    expan.plot_amplitude_vs_rmse(df, target_samples, exp.sweep_name, exp.sweep_dir, show=show_all)
    expan.plot_masks(algo_sweep.algo_masks, expan.plot_amplitude_vs_rmse, df, target_samples, exp.sweep_name, exp.sweep_dir, show=show_all)
    data_io.hoard_experiment_results(exp.sweep_name, results, df, exp.sweep_dir)

def run_hyperparameter_optimization():
    """run experiments to determine the best experimental parameters"""
    # take best algorithm
    # then also increase the z_ops to see if the weight-range-to-rmse curve flattens
    # same for rmse vs n_osc
    # also for frequency band


@data_analysis.print_time
def main():
    # Python
    if True:
        generator_args = params_python.py_rand_args_uniform
        m_target = meta_target.MetaTargetSample(generator_args, "magpie", params_target.DevSet.MAGPIE.value)
        algo_sweep_test = param_util.init_algo_sweep(m_target, generator_args, max_z_ops=params_python.MAX_Z_OPS, m_averages=params_python.M_AVERAGES)

        # qualitative_algo_sweep(algo_sweep_test, m_target, visual=True)
        produce_all_results(algo_sweep_test, m_target.samples, generator_args)

    # SpiPy
    if False:
        rand_args = params_hybrid.spice_rand_args_uniform
        m_target = meta_target.MetaTargetTime(rand_args, "magpie", params_target.DevSet.MAGPIE.value)
        print(m_target.samples)

        # scale the number of samples in the target to the number of samples produced by spice
        signal_generator = gen_signal_spipy.SpipySignalGenerator()
        spice_samples = signal_generator.estimate_number_of_samples(rand_args)
        m_target.adjust_samples(spice_samples)

        algo_sweep = param_util.init_algo_sweep(m_target, rand_args, sig_generator=signal_generator, max_z_ops=5e3, m_averages=1)

        qualitative_algo_sweep(algo_sweep, m_target, visual=True)
        # produce_all_results(algo_sweep, m_target.samples, rand_args)

if __name__ == "__main__":
    main()

# IDEA: maybe it would be cool to write an algorithm for self-adjustment of the correct number of oscillators