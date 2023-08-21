"""
This module implements the Experimenteur class.

The Experimenteur class runs experiments.
"""

import copy
from dataclasses import fields
from typing import Final, Iterable, List, Tuple, Union
from multiprocessing import Pool, cpu_count
from functools import partial

import data_analysis
import experiment_analysis as expan
import gen_signal_args_types as party
import sweep_types as sweety
import result_types as resty
import algo
import meta_target
import const
import data_io
import shared_params_target
import sample
import gen_signal_python
import gen_signal_spipy
import sweep_builder
import algo_args_bundle
import algo_args_type

if const.TEST_PARAMS:
    print("Import test parameters.")
    import params_hybrid_test as hybrid_parameters
else:
    print("Import production parameters.")
    import params_hybrid as hybrid_parameters

import numpy as np


class Experimenteur:
    """Controls the execution of experiments and aggregates results."""

    def __init__(
        self,
        experiment_name: str = "",
        mp: bool = const.MULTIPROCESSING,
        clean_work_dir: bool = False,
        show_plots: bool = False,
    ) -> None:
        """
        Initialize the Experimenteur class.

        args:
            mp: if true use multiple CPUs for processing
            clean_work_dir: if true delete all files in the work directory
        """
        self.mp = mp
        self.cpu_count = cpu_count()
        if clean_work_dir:
            data_io.clean_dir(const.WRITE_DIR)
        experiment_name = (
            "quantitative_experiment"
            if experiment_name == ""
            else f"quantitative_experiment_{experiment_name}"
        )
        self.work_dir = data_io.find_dir_name(
            const.WRITE_DIR, experiment_name
        )  # directory in which to write all results
        self.sweep_dir = (
            None  # directory in which to write results of a sweep for an independent variable
        )
        self.sweep_name = None  # name of the sweep for an independent variable
        self.show_plots = show_plots

    @staticmethod
    def run_qualitative_algo_sweep(
        sweep_bundle: sweety.SweepBundle, m_target: meta_target.MetaTarget
    ) -> None:
        """
        Perform an experiment to compare multiple algorithms but don't collect results over multiple runs or average.

        Plots the best sample for each algorithm against the target.
        """
        local_target = copy.deepcopy(m_target)
        write_dir = data_io.find_dir_name(
            const.WRITE_DIR, f"qualitative_{sweep_bundle.signal_generator.__class__.__name__}"
        )
        for awa in sweep_bundle.algo_sweep.algo_with_args:
            awa: algo_args_bundle.AlgoWithArgs
            search_alg = awa.Algo(awa.algo_args)
            best_sample, z_ops = search_alg.search()
            sample.evaluate_prediction(
                best_sample,
                local_target,
                z_ops,
                search_alg.__class__.__name__,
                sweep_bundle.signal_generator.__class__.__name__,
                write_dir=write_dir,
            )

    def extract_weights(self, samples: Iterable) -> np.ndarray:
        """extract and concatenate the weights of the samples in the iterable to a single array"""
        weights = [s.weights for s in samples]
        weights = np.array(weights)
        flat_weights = weights.flatten()
        n_oscillators = len(samples[0].weights)
        assert flat_weights.shape == (len(samples) * n_oscillators,)
        return flat_weights

    def extract_frequencies(self, samples: Iterable, sample_spacing: float) -> np.ndarray:
        """extract and concatenate the fundamental frequencies of the samples in the iterable to a single array"""
        f0_li = []
        for samp in samples:
            sm = samp.signal_matrix
            for osc in range(sm.shape[0]):
                f0_li.append(data_analysis.get_freq_from_fft_v2(sm[osc, :], sample_spacing))

        frequencies = np.array(f0_li)
        n_oscillators = len(samples[0].weights)
        assert frequencies.shape == (len(samples) * n_oscillators,)
        return frequencies

    def extract_meta_data(self, algo_args: algo_args_type.AlgoArgs, m_averages: int) -> dict:
        """convert an algorithm's meta data to a dictionary"""
        meta_dict = dict()
        meta_dict["target_name"] = algo_args.meta_target.name
        meta_dict["max_z_ops"] = algo_args.max_z_ops
        meta_dict["n_osc"] = algo_args.rand_args.n_osc
        meta_dict["sampling_rate"] = algo_args.meta_target.sampling_rate
        meta_dict["m_averages"] = m_averages
        return meta_dict

    def init_dict(self):
        """
        initialize a dictionary to store experiment results

        data contains the optimized oscillator ensemble and its parameters
        meta contains the search parameters and additional meta data such as names
        """
        return {"data": dict(), "meta": dict()}

    def run_osc_attribute_dist_sweep(
        self, sweep_bundle: sweety.SweepBundle, base_args: party.UnionRandArgs, attr_selector: str
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Run an experiment with the same parameters (base args) but varying algorithms to collect the distribution of an attribute across multiple runs

        args:
            sweep_bundle: contains the signal generator and the algo sweep
            base_args: the base arguments for the random signal generator
            attr_selector: what attribute to track in [weights, freq]
        """
        results = dict()
        for awa in sweep_bundle.algo_sweep.algo_with_args:
            awa: algo_args_bundle.AlgoWithArgs
            search_alg = awa.Algo(awa.algo_args)
            m_ensembles, z_ops = self.invoke_search(
                search_alg, sweep_bundle.algo_sweep
            )  # m corresponds to number of runs

            algo_name = search_alg.__class__.__name__
            results[algo_name] = self.init_dict()
            results[algo_name]["meta"] = self.extract_meta_data(
                awa.algo_args, sweep_bundle.algo_sweep.m_averages
            )

            # add distribution data to results dictionary
            if attr_selector == "weights":
                results[algo_name]["data"]["weights"] = self.extract_weights(m_ensembles)
            elif attr_selector == "freq":
                results[algo_name]["data"]["freq"] = self.extract_frequencies(
                    m_ensembles, base_args.get_sample_spacing()
                )
            else:
                raise ValueError(f"attr_selector must be in [weights, freq] but is {attr_selector}")

        return results

    def set_sweep_name_and_dir(self, sweep_name: str) -> None:
        """set the name and directory of the sweep for the next experiment"""
        self.sweep_name = sweep_name
        self.sweep_dir = data_io.find_dir_name(self.work_dir, sweep_name)

    def invoke_search(self, search_alg: algo.SearchAlgo, algo_sweep: sweety.AlgoSweep) -> Iterable:
        """call an algorithm's search function for a given number of times"""
        if self.mp:
            with Pool(self.cpu_count) as p:
                mapfunc = partial(search_alg.search, mp=self.mp)  # pass mp in kwargs to search
                raw_result = p.map(mapfunc, range(algo_sweep.m_averages))
                best_samples, z_ops = zip(*raw_result)
        else:
            raw_result = map(search_alg.search, range(algo_sweep.m_averages))
            best_samples, z_ops = zip(*raw_result)
        return best_samples, z_ops

    def average_result(
        self,
        samples: Iterable,
        m_z_ops: Iterable,
        search_alg: algo.SearchAlgo,
        algo_sweep: sweety.AlgoSweep,
    ) -> resty.ResultSweep:
        """produce mean, stddev for rmse and z-ops over a list of samples (fit signal ensembles)"""
        m_rmse = [s.rmse for s in samples]
        m_rmse = np.array(m_rmse)
        m_z_ops = np.array(m_z_ops)
        mean_rmse, mean_z_ops = np.mean(m_rmse), np.mean(
            m_z_ops
        )  # map has effects and two functions per map are too ugly
        std_rmse, std_z_ops = np.std(m_rmse), np.std(m_z_ops)
        return resty.ResultSweep(
            search_alg.__class__.__name__,
            search_alg.get_algo_args(),
            mean_rmse,
            std_rmse,
            mean_z_ops,
            std_z_ops,
            algo_sweep.m_averages,
        )

    def run_algo_sweep(self, algo_sweep: sweety.AlgoSweep) -> List[resty.ResultSweep]:
        """run an experiment comparing multiple algorithms on their rmse and operations"""
        results = list()
        for awa in algo_sweep.algo_with_args:
            awa: algo_args_bundle.AlgoWithArgs
            search_alg = awa.Algo(awa.algo_args)  # BUG: algo args are wrong here
            best_samples, z_ops = self.invoke_search(search_alg, algo_sweep)
            result = self.average_result(best_samples, z_ops, search_alg, algo_sweep)
            results.append(result)
        return results

    def run_rand_args_sweep(
        self,
        algo_sweep: sweety.AlgoSweep,
        sweep_args: Union[sweety.ConstTimeSweep, sweety.ExpoTimeSweep],
        base_args: party.UnionRandArgs,
    ) -> resty.ResultSweep:
        """
        run an experiment across algorithms and a dependent variable

        results are averaged over multiple runs

        args:
            algo_sweep: a list of algorithms and algorithm arguments, the algorithm arguments will be modified
            sweep_args: an attribute within a rand_args type, for each attribute a list of values is tested
        """
        print("sweeping with", sweep_args.__class__.__name__)
        results = []
        for val_schedule in fields(sweep_args):  # for example frequency distribution
            for awa in algo_sweep.algo_with_args:  # for example monte carlo search
                awa: algo_args_bundle.AlgoWithArgs
                for val in getattr(
                    sweep_args, val_schedule.name
                ):  # for example normal vs uniform frequency distribution
                    temp_args = copy.deepcopy(base_args)  # init/reset temporary rand_args
                    setattr(
                        temp_args, val_schedule.name, val
                    )  # for example, for field frequency in base_args set value 10 Hz
                    if val_schedule.name == "n_osc":  # when n_osc changes
                        temp_args.weight_dist.n = val  #    update n also in weight_dist
                    awa.algo_args.rand_args = temp_args
                    f_algo_args: Final = copy.deepcopy(awa.algo_args)

                    search_alg: algo.SearchAlgo = awa.Algo(f_algo_args)
                    samples, z_ops = self.invoke_search(search_alg, algo_sweep)
                    result = self.average_result(samples, z_ops, search_alg, algo_sweep)
                    results.append(result)
        # TODO: flush and pickle results
        return results

    # TODO: refactor
    # def alternative_run_rand_args_sweep():
    #     for val in algo_sweep.dv_values:
    #         for awa in algo_sweep.algo_with_args:
    #             temp_args = copy.deepcopy(algo_args.rand_args)
    #             setattr(temp_args, algo_sweep.dv_name, val)
    #             if algo_sweep.dv_name == "n_osc":
    #                 temp_args.weight_dist.n = val

    #             awa.algo_args.rand_args = temp_args
    #             f_algo_args: Final = copy.deepcopy(awa.algo_args)
    #             search_alg: algo.SearchAlgo = awa.Algo(f_algo_args)

    def run_samples_sweep(
        self, sweep_bundle: sweety.SweepBundle, base_args: party.UnionRandArgs, algo_selector: str
    ) -> resty.ResultSweep:
        """run all algorithms with targets of varying lengths (number of samples)"""
        print("sweeping with", sweep_bundle.num_samples_sweep.__class__.__name__)
        results = list()
        for s in sweep_bundle.num_samples_sweep.samples:
            temp_args = copy.deepcopy(base_args)
            temp_args.samples = s  # inject samples into rand_args
            m_target = meta_target.MetaTargetSample(
                temp_args, "magpie", shared_params_target.DevSet.MAGPIE.value
            )
            algo_sweep = sweep_builder.build_algo_sweep(
                sweep_bundle.signal_generator,
                temp_args,
                m_target,
                sweep_bundle.max_z_ops,
                sweep_bundle.m_averages,
                algo_selector,
            )
            results += self.run_algo_sweep(algo_sweep)
        return results

    def run_duration_sweep(
        self, sweep_bundle: sweety.SweepBundle, base_args: party.UnionRandArgs, target_selector: str
    ) -> List[resty.ResultSweep]:
        """run all algorithms with targets of varying durations"""
        print("sweeping with", sweep_bundle.duration_sweep.__class__.__name__)
        
        if sweep_bundle.duration_sweep is None:
            return []

        # TODO: build external interface that allows uniform access the generator args
        # for example access to time_start and time_stop for spice and duration for python
        # e.g. let generator_args be an abstract class with a method get_duration() that returns the duration
        def inject_duration(generator_args: party.UnionRandArgs) -> party.UnionRandArgs:
            """inject duration into rand_args"""
            temp_args = copy.deepcopy(generator_args)
            if isinstance(generator_args, party.SpiceSumRandArgs):
                assert temp_args.time_start == 0, "as implemented, time_start must be 0"
                temp_args.time_stop = d
            elif isinstance(generator_args, party.PythonSignalRandArgs):
                temp_args.duration = d
            return temp_args

        results = list()
        for d in sweep_bundle.duration_sweep.duration:
            temp_args = inject_duration(base_args)
            m_target = shared_params_target.select_target_by_string(
                target_selector, temp_args, hybrid_parameters.SYNTH_FREQ
            )
            algo_sweep = sweep_builder.build_algo_sweep(
                sweep_bundle.signal_generator,
                temp_args,
                m_target,
                sweep_bundle.max_z_ops,
                sweep_bundle.m_averages,
            )
            results += self.run_algo_sweep(algo_sweep)
        return results

    def run_z_ops_sweep(
        self,
        sweep_bundle: sweety.UnionSweepBundle,
        base_args: party.UnionRandArgs,
        algo_selector: str,
    ) -> List[resty.ResultSweep]:
        """run all algorithms with different numbers of z-operations, corresponding to more extensive search"""
        print("sweeping with", sweep_bundle.z_ops_sweep.__class__.__name__)
        local_args = copy.deepcopy(base_args)
        meta_target = sweep_bundle.algo_sweep.algo_with_args[0].algo_args.meta_target
        results = list()
        for z_ops in sweep_bundle.z_ops_sweep.max_z_ops:
            algo_sweep = sweep_builder.build_algo_sweep(
                sweep_bundle.signal_generator,
                local_args,
                meta_target,
                z_ops,
                sweep_bundle.m_averages,
                algo_selector,
            )
            results += self.run_algo_sweep(algo_sweep)
        return results

    def run_target_sweep(
        self,
        target_sweep: sweety.TargetSweep,
        algo_selector: str,
    ) -> List[resty.ResultSweep]:
        """run an experiment comparing multiple targets on their rmse and operations"""
        print("sweeping with", target_sweep.__class__.__name__)
        results = list()
        for target in target_sweep.targets:
            algo_sweep = sweep_builder.build_algo_sweep(
                target_sweep.signal_generator,
                target_sweep.rand_args,
                target,
                target_sweep.max_z_ops,
                target_sweep.m_averages,
                algo_selector,
            )
            results += self.run_algo_sweep(algo_sweep)
        return results

    def run_all_experiments(
        self,
        sweep_bundle: sweety.UnionSweepBundle,
        target_samples: int,
        base_rand_args: party.UnionRandArgs,
        selector: str,
        target_selector: str | None = None,
        algo_selector: str | None = None,
        load_results: str = "",
    ) -> None:
        """run all experiments and plot results

        args:
            selector: experiment to run
            target_selector: target to use for experiments that require initializing a target
            algo_selector: algorithms to use for experiments
            load_results: load results from a previous run, pass the name of the sweep to load
        """

        def invoke_target_sweep():
            """run an experiment with varying target functions"""
            self.set_sweep_name_and_dir("targets_vs_rmse")
            results = self.run_target_sweep(sweep_bundle.target_sweep, algo_selector)
            df = expan.conv_results_to_pd(results)
            expan.analyze_targets_vs_rmse(df, self.sweep_name, self.sweep_dir, show=self.show_plots)
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_target_freq_sweep():
            """run an experiment with the same target function at different frequencies"""
            self.set_sweep_name_and_dir("target_freq_vs_rmse")
            results = self.run_target_sweep(sweep_bundle.target_freq_sweep, algo_selector)
            df = expan.conv_results_to_pd(results)
            expan.analyze_targets_vs_rmse(df, self.sweep_name, self.sweep_dir, show=self.show_plots)
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_n_osc_sweep():
            """run an experiment with different numbers of oscillators"""
            self.set_sweep_name_and_dir("n_osc_vs_rmse")
            results = self.run_rand_args_sweep(
                sweep_bundle.algo_sweep, sweep_bundle.n_osc_sweep, base_rand_args
            )
            df = expan.conv_results_to_pd(results)
            expan.plot_n_vs_rmse(
                df, target_samples, self.sweep_name, self.sweep_dir, show=self.show_plots
            )
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_n_vs_rmse,
                df,
                target_samples,
                self.sweep_name,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_z_ops_sweep():
            """run an experiment where the number of perturbations (Z) to an oscillator ensemble is varied"""
            self.set_sweep_name_and_dir("z_ops_vs_rmse")
            if load_results != "z_ops":
                results = self.run_z_ops_sweep(sweep_bundle, base_rand_args, algo_selector)
                df = expan.conv_results_to_pd(results)
            else:
                results = None
                df = data_io.load_experiment_results("z_ops")
            expan.plot_z_vs_rmse(
                df, target_samples, self.sweep_name, self.sweep_dir, show=self.show_plots
            )
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_z_vs_rmse,
                df,
                target_samples,
                self.sweep_name,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_samples_sweep():
            """run an experiment where the number of the samples in the target is varied"""
            self.set_sweep_name_and_dir("samples_vs_rmse")
            results = self.run_samples_sweep(sweep_bundle, base_rand_args, algo_selector)
            df = expan.conv_results_to_pd(results)
            expan.plot_samples_vs_rmse(df, self.sweep_name, self.sweep_dir, show=self.show_plots)
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_samples_vs_rmse,
                df,
                self.sweep_name,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_duration_sweep():
            """run an experiment where the duration of the target is varied"""
            self.set_sweep_name_and_dir("duration_vs_rmse")
            results = self.run_duration_sweep(sweep_bundle, base_rand_args, target_selector)
            df = expan.conv_results_to_pd(results)
            expan.plot_duration_vs_rmse(df, self.sweep_name, self.sweep_dir, show=self.show_plots)
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_duration_vs_rmse,
                df,
                self.sweep_name,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_freqs_sweep():
            """run an experiment where the frequency diversity of oscillators is varied"""
            freq_sweeps = [sweep_bundle.freq_sweep_from_zero, sweep_bundle.freq_sweep_around_vo2]
            freq_sweep_names = ["freq_range_from_zero", "freq_range_around_vo2"]
            for freq_sweep, freq_sweep_name in zip(freq_sweeps, freq_sweep_names):
                self.set_sweep_name_and_dir(freq_sweep_name)
                results = self.run_rand_args_sweep(
                    sweep_bundle.algo_sweep, freq_sweep, base_rand_args
                )
                df = expan.conv_results_to_pd(results)
                expan.plot_freq_range_vs_rmse(
                    df, target_samples, freq_sweep_name, self.sweep_dir, show=self.show_plots
                )
                expan.plot_masks(
                    sweep_bundle.algo_sweep.algo_masks,
                    expan.plot_freq_range_vs_rmse,
                    df,
                    target_samples,
                    freq_sweep_name,
                    self.sweep_dir,
                    show=self.show_plots,
                )
                data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_weight_sweep():
            """run an experiment where the weight (gain) diversity of the oscillators is varied, also known as dynamic range"""
            self.set_sweep_name_and_dir("weight_range_vs_rmse")
            results = self.run_rand_args_sweep(
                sweep_bundle.algo_sweep, sweep_bundle.weight_sweep, base_rand_args
            )
            df = expan.conv_results_to_pd(results)
            expan.plot_weight_range_vs_rmse(
                df, target_samples, self.sweep_dir, show=self.show_plots
            )
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_weight_range_vs_rmse,
                df,
                target_samples,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_phase_sweep():
            """run an experiment where the phase diversity of the oscillators is varied"""
            self.set_sweep_name_and_dir("phase_range_vs_rmse")
            results = self.run_rand_args_sweep(
                sweep_bundle.algo_sweep, sweep_bundle.phase_sweep, base_rand_args
            )
            df = expan.conv_results_to_pd(results)
            expan.plot_phase_range_vs_rmse(df, target_samples, self.sweep_dir, show=self.show_plots)
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_phase_range_vs_rmse,
                df,
                target_samples,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_offset_sweep():
            """run an experiment where the offset diversity of the oscillator ensemble is varied"""
            self.set_sweep_name_and_dir("offset_range_vs_rmse")
            results = self.run_rand_args_sweep(
                sweep_bundle.algo_sweep, sweep_bundle.offset_sweep, base_rand_args
            )
            df = expan.conv_results_to_pd(results)
            expan.plot_offset_range_vs_rmse(
                df, target_samples, self.sweep_dir, show=self.show_plots
            )
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_offset_range_vs_rmse,
                df,
                target_samples,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_amplitude_sweep():
            """run an experiment where the amplitude diversity of oscillators is varied"""
            self.set_sweep_name_and_dir("amplitude_vs_rmse")
            results = self.run_rand_args_sweep(
                sweep_bundle.algo_sweep, sweep_bundle.amplitude_sweep, base_rand_args
            )
            df = expan.conv_results_to_pd(results)
            expan.plot_amplitude_vs_rmse(
                df, target_samples, self.sweep_name, self.sweep_dir, show=self.show_plots
            )
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_amplitude_vs_rmse,
                df,
                target_samples,
                self.sweep_name,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_resistor_sweep():
            """run an experiment where the resistor diversity of oscillators is varied"""
            self.set_sweep_name_and_dir("resistor_vs_rmse")
            results = self.run_rand_args_sweep(
                sweep_bundle.algo_sweep, sweep_bundle.resistor_sweep, base_rand_args
            )
            df = expan.conv_results_to_pd(results)
            expan.plot_resistor_range_vs_rmse(
                df, target_samples, self.sweep_dir, show=self.show_plots
            )
            expan.plot_masks(
                sweep_bundle.algo_sweep.algo_masks,
                expan.plot_resistor_range_vs_rmse,
                df,
                target_samples,
                self.sweep_dir,
                show=self.show_plots,
            )
            data_io.hoard_experiment_results(self.sweep_name, results, df, self.sweep_dir)

        def invoke_gain_dist_sweep():
            """run an experiment where the distribution of gains (weights) of oscillators in an ensemble is plotted across multiple algorithms"""
            self.set_sweep_name_and_dir("gain_dist")
            if load_results != "gain_dist":
                results = self.run_osc_attribute_dist_sweep(sweep_bundle, base_rand_args, "weights")
                expan.plot_multi_weight_hist(results, self.sweep_dir, show=self.show_plots)
                data_io.hoard_experiment_results(self.sweep_name, results, None, self.sweep_dir)
            else:
                results = data_io.load_experiment_results("gain_dist_results.pickle")
                expan.plot_multi_weight_hist_2x2(results, self.sweep_dir, show=self.show_plots)

        def invoke_freq_dist_sweep():
            """run an experiment where the distribution of frequencies of oscillators in an ensemble is plotted across multiple algorithms"""
            self.set_sweep_name_and_dir("freq_dist")
            if load_results != "freq_dist":
                results = self.run_osc_attribute_dist_sweep(sweep_bundle, base_rand_args, "freq")
                expan.plot_multi_freq_hist(results, self.sweep_dir, show=self.show_plots)
                data_io.hoard_experiment_results(self.sweep_name, results, None, self.sweep_dir)
            else:
                results = data_io.load_experiment_results("freq_dist_results.pickle")
                expan.plot_multi_freq_hist_2x2(results, self.sweep_dir, show=self.show_plots)

        @data_analysis.print_time
        def invoke_python_generator_sweeps():
            if selector in ["all", "target"]:
                invoke_target_sweep()
            if selector in ["all", "n_osc"]:
                invoke_n_osc_sweep()
            if selector in ["all", "z_ops"]:
                invoke_z_ops_sweep()
            if selector in ["all", "samples"]:
                invoke_samples_sweep()
            if selector in ["all", "frequency"]:
                invoke_freqs_sweep()
            if selector in ["all", "weight"]:
                invoke_weight_sweep()
            if selector in ["all", "phase"]:
                invoke_phase_sweep()
            if selector in ["all", "offset"]:
                invoke_offset_sweep()
            if selector in ["all", "amplitude"]:
                invoke_amplitude_sweep()
            if selector in ["all", "gain_dist"]:
                invoke_gain_dist_sweep()
            if selector in ["all", "freq_dist"]:
                invoke_freq_dist_sweep()
            # TODO
            # invoke_duration_sweep()

        @data_analysis.print_time
        def invoke_hybrid_generator_sweeps():
            if selector in ["all", "target"]:
                invoke_target_sweep()
            if selector in ["all", "target_freq"]:
                invoke_target_freq_sweep()
            if selector in ["all", "n_osc"]:
                invoke_n_osc_sweep()
            if selector in ["all", "z_ops"]:
                invoke_z_ops_sweep()
            if selector in ["all", "duration"]:
                invoke_duration_sweep()
            if selector in ["all", "resistor"]:
                invoke_resistor_sweep()
            if selector in ["all", "weight"]:
                invoke_weight_sweep()
            if selector in ["all", "phase"]:
                invoke_phase_sweep()
            if selector in ["all", "offset"]:
                invoke_offset_sweep()
            if selector in ["all", "gain_dist"]:
                invoke_gain_dist_sweep()
            if selector in ["all", "freq_dist"]:
                invoke_freq_dist_sweep()

        if isinstance(sweep_bundle.signal_generator, gen_signal_python.PythonSigGen):
            print("Start experiment with Python signal generator")
            invoke_python_generator_sweeps()
        elif isinstance(sweep_bundle.signal_generator, gen_signal_spipy.SpipySignalGenerator):
            print("Start experiment with hybrid signal generator")
            invoke_hybrid_generator_sweeps()
        else:
            raise ValueError("Unknown signal generator type")

    def run_hyperparameter_optimization(self):
        """run experiments to determine the best experimental parameters"""
        raise NotImplementedError("TODO")
        # take best algorithm
        # then also increase the z_ops to see if the weight-range-to-rmse curve flattens
        # same for rmse vs n_osc
        # also for frequency band
