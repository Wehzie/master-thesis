"""
This module bundles sweep parameters into a set of algorithm sweeps accessible through an enum.
"""

from typing import Union
import const
import sweep_types as sweety
import meta_target
import gen_signal_args_types as party
import gen_signal_python
import gen_signal_spipy
import shared_params_mask
import shared_params_algos
if const.TEST_PARAMS:
    import params_python_test as params_python
    import params_hybrid_test as params_hybrid
else:
    import params_python
    import params_hybrid


def build_algo_sweep(
sig_generator: gen_signal_python.PythonSigGen,
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
        max_z_ops=params_python.MAX_Z_OPS,
        m_averages=params_python.M_AVERAGES,
        algo_selector=algo_selector,
    )

    python_sweep_bundle = sweety.PythonSweepBundle(
        description="Experiments with the python signal generator",
        signal_generator=sig_generator,
        generator_args=generator_args,
        max_z_ops=params_python.MAX_Z_OPS,
        m_averages=params_python.M_AVERAGES,
        
        algo_sweep=algo_sweep,

        target_sweep=params_python.target_sweep_samples,
        # TODO:
        # target_sweep_time = params_python.target_sweep_time,
        duration_sweep=params_python.duration_sweep,
        n_osc_sweep=params_python.n_osc_sweep,
        z_ops_sweep=params_python.z_ops_sweep,
        num_samples_sweep=params_python.num_samples_sweep,
        freq_sweep_from_zero=params_python.freq_sweep_from_zero,
        freq_sweep_around_vo2=params_python.freq_sweep_around_vo2,
        amplitude_sweep=params_python.amplitude_sweep,
        weight_sweep=params_python.weight_sweep,
        phase_sweep=params_python.phase_sweep,
        offset_sweep=params_python.offset_sweep,
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
        max_z_ops=params_hybrid.MAX_Z_OPS,
        m_averages=params_hybrid.M_AVERAGES,
        algo_selector=algo_selector,
    )

    hybrid_sweep_bundle = sweety.HybridSweepBundle(
        description="Experiments with the hybrid signal generator",
        signal_generator=sig_generator,
        generator_args=generator_args,
        max_z_ops=params_python.MAX_Z_OPS,
        m_averages=params_python.M_AVERAGES,

        algo_sweep=algo_sweep,

        target_sweep=params_hybrid.target_sweep,
        n_osc_sweep=params_hybrid.n_osc_sweep,
        z_ops_sweep=params_hybrid.z_ops_sweep,
        duration_sweep=params_hybrid.duration_sweep,
        resistor_sweep=params_hybrid.resistor_sweep,
        weight_sweep=params_hybrid.weight_sweep,
        phase_sweep=params_hybrid.phase_sweep,
        offset_sweep=params_hybrid.offset_sweep,
    )
    return hybrid_sweep_bundle