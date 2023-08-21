"""This module bundles sweep parameters into a set of algorithm sweeps accessible through an enum."""

from typing import Union
import const
import sweep_types as sweety
import meta_target
import gen_signal_args_types as party
import gen_signal
import gen_signal_python
import gen_signal_spipy
import shared_params_mask
import shared_params_algos

if const.TEST_PARAMS:
    print("Import test parameters.")
    import params_python_test as python_parameters
    import params_hybrid_test as hybrid_parameters
else:
    print("Import production parameters.")
    import params_python as python_parameters
    import params_hybrid as hybrid_parameters


def build_algo_sweep(
    sig_generator: gen_signal.SignalGenerator,
    generator_args: party.UnionRandArgs,
    meta_target: meta_target.MetaTarget,
    max_z_ops: Union[int, None],
    m_averages: int,
    algo_selector: str = "all",
) -> sweety.AlgoSweep:
    """Build an AlgoSweep with a set of algorithms."""

    algo_with_args = shared_params_algos.bundle_algos_with_args(
        sig_generator=sig_generator,
        generator_args=generator_args,
        meta_target=meta_target,
        max_z_ops=max_z_ops,
        selector=algo_selector,
    )
    algo_sweep = sweety.AlgoSweep(
        algo_with_args=algo_with_args,
        m_averages=m_averages,
        algo_masks=shared_params_mask.algo_masks,
    )
    return algo_sweep


def bundle_python_sweep(
    sig_generator: gen_signal_python.PythonSigGen,
    generator_args: party.UnionRandArgs,
    meta_target: meta_target.MetaTarget,
    algo_selector: str = "all",
) -> sweety.PythonSweepBundle:
    """Bundle an AlgoSweep with a set of secondary independent variables to be used with the Python signal generator."""

    algo_sweep = build_algo_sweep(
        sig_generator=sig_generator,
        generator_args=generator_args,
        meta_target=meta_target,
        max_z_ops=python_parameters.MAX_Z_OPS,
        m_averages=python_parameters.M_AVERAGES,
        algo_selector=algo_selector,
    )

    python_sweep_bundle = sweety.PythonSweepBundle(
        description="Experiments with the python signal generator",
        signal_generator=sig_generator,
        generator_args=generator_args,
        max_z_ops=python_parameters.MAX_Z_OPS,
        m_averages=python_parameters.M_AVERAGES,
        algo_sweep=algo_sweep,
        target_sweep=python_parameters.target_sweep_samples,
        # TODO:
        # target_sweep_time = python_parameters.target_sweep_time,
        duration_sweep=python_parameters.duration_sweep,
        n_osc_sweep=python_parameters.n_osc_sweep,
        z_ops_sweep=python_parameters.z_ops_sweep,
        num_samples_sweep=python_parameters.num_samples_sweep,
        freq_sweep_from_zero=python_parameters.freq_sweep_from_zero,
        freq_sweep_around_vo2=python_parameters.freq_sweep_around_vo2,
        amplitude_sweep=python_parameters.amplitude_sweep,
        weight_sweep=python_parameters.weight_sweep,
        phase_sweep=python_parameters.phase_sweep,
        offset_sweep=python_parameters.offset_sweep,
    )
    return python_sweep_bundle


def bundle_hybrid_sweep(
    sig_generator: gen_signal_spipy.SpipySignalGenerator,
    generator_args: party.UnionRandArgs,
    meta_target: meta_target.MetaTarget,
    algo_selector: str = "all",
) -> sweety.HybridSweepBundle:
    """Bundle an AlgoSweep with a set of secondary independent variables to be used with the hybrid signal generator."""

    algo_sweep = build_algo_sweep(
        sig_generator=sig_generator,
        generator_args=generator_args,
        meta_target=meta_target,
        max_z_ops=hybrid_parameters.MAX_Z_OPS,
        m_averages=hybrid_parameters.M_AVERAGES,
        algo_selector=algo_selector,
    )

    hybrid_sweep_bundle = sweety.HybridSweepBundle(
        description="Experiments with the hybrid signal generator",
        signal_generator=sig_generator,
        generator_args=generator_args,
        max_z_ops=hybrid_parameters.MAX_Z_OPS,
        m_averages=hybrid_parameters.M_AVERAGES,
        algo_sweep=algo_sweep,
        target_sweep=hybrid_parameters.target_sweep,
        target_freq_sweep=hybrid_parameters.target_freq_sweep,
        n_osc_sweep=hybrid_parameters.n_osc_sweep,
        z_ops_sweep=hybrid_parameters.z_ops_sweep,
        duration_sweep=hybrid_parameters.duration_sweep,
        resistor_sweep=hybrid_parameters.resistor_sweep,
        weight_sweep=hybrid_parameters.weight_sweep,
        phase_sweep=hybrid_parameters.phase_sweep,
        offset_sweep=hybrid_parameters.offset_sweep,
    )
    return hybrid_sweep_bundle
