"""
This module bundles all algorithms into a single module.

A function is provided to produce a list of algorithms with partially shared arguments. 
"""

from typing import List, Union

import algo_args_type as algarty
import algo_args_bundle as algabun
import algo
import algo_las_vegas as alave
import algo_monte_carlo as almoca
import algo_evolution as alevo
import algo_gradient as algra
import algo_mcmc as almcmc
import gen_signal_args_types as party
import meta_target
import gen_signal


def bundle_algos_with_args(
    sig_generator: gen_signal.SignalGenerator,
    generator_args: party.UnionRandArgs,
    meta_target: meta_target.MetaTarget,
    max_z_ops: Union[int, None] = None,
    selector: str = "all",
) -> List[algabun.AlgoWithArgs]:
    """return a list of algorithms and their arguments"""
    one_shot_algos = [
        algabun.AlgoWithArgs(
            almoca.MCOneShot,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
        algabun.AlgoWithArgs(
            almoca.MCOneShotWeight,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
    ]
    exploit_algos = [
        algabun.AlgoWithArgs(
            almoca.MCExploit,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=1),
        ),
        algabun.AlgoWithArgs(
            almoca.MCExploitJ10,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=10),
        ),
        algabun.AlgoWithArgs(
            almoca.MCExploitWeight,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=1),
        ),
        algabun.AlgoWithArgs(
            almoca.MCExploitNeighborWeight,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=1),
        ),
        algabun.AlgoWithArgs(
            almoca.MCExploitDecoupled,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=1),
        ),
        algabun.AlgoWithArgs(
            almoca.MCExploitFast,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=1),
        ),
    ]
    grow_shrink_algos = [
        algabun.AlgoWithArgs(
            almoca.MCGrowShrink,
            algarty.AlgoArgs(
                sig_generator,
                generator_args,
                meta_target,
                max_z_ops,
                j_replace=1,
                l_damp_prob=0.5,
                h_damp_fac=0.5,
            ),
        ),
        algabun.AlgoWithArgs(
            almoca.MCDampen,
            algarty.AlgoArgs(
                sig_generator, generator_args, meta_target, max_z_ops, j_replace=1, h_damp_fac=0.5
            ),
        ),
        algabun.AlgoWithArgs(
            almoca.MCPurge,
            algarty.AlgoArgs(
                sig_generator, generator_args, meta_target, max_z_ops, j_replace=1, h_damp_fac=0
            ),
        ),
    ]
    oscillator_anneal_algos = [
        algabun.AlgoWithArgs(
            almoca.MCOscillatorAnneal,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
        algabun.AlgoWithArgs(
            almoca.MCOscillatorAnnealWeight,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
        algabun.AlgoWithArgs(
            almoca.MCOscillatorAnnealLog,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
        algabun.AlgoWithArgs(
            almoca.MCOscillatorAnnealLogWeight,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
    ]
    las_vegas_algos = [
        algabun.AlgoWithArgs(
            alave.LasVegas,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
        algabun.AlgoWithArgs(
            alave.LasVegasWeight,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
    ]
    population_algos = [
        algabun.AlgoWithArgs(
            alevo.DifferentialEvolution,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
    ]
    mcmc_algos = [
        algabun.AlgoWithArgs(
            almcmc.MCExploitErgodic,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=1),
        ),
        algabun.AlgoWithArgs(
            almcmc.MCExploitAnneal,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=1),
        ),
        algabun.AlgoWithArgs(
            almcmc.MCExploitAnnealWeight,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops, j_replace=1),
        ),
        algabun.AlgoWithArgs(
            almcmc.BasinHopping,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
        algabun.AlgoWithArgs(
            almcmc.ScipyAnneal,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
        algabun.AlgoWithArgs(
            almcmc.ScipyDualAnneal,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
    ]
    gradient_algos = [
        algabun.AlgoWithArgs(
            algra.LinearRegression,
            algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
        ),
    ]

    if selector == "test":
        out_algos = [
            algabun.AlgoWithArgs(
                almoca.MCOneShotWeight,
                algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
            ),
        ]
    elif selector == "all":
        out_algos = (
            one_shot_algos
            + gradient_algos
            + exploit_algos
            + las_vegas_algos
            + population_algos
            + mcmc_algos
            + oscillator_anneal_algos
            + grow_shrink_algos
        )
    elif selector == "best":
        out_algos = [
            algabun.AlgoWithArgs(
                almoca.MCExploitWeight,
                algarty.AlgoArgs(
                    sig_generator, generator_args, meta_target, max_z_ops, j_replace=1
                ),
            ),
            algabun.AlgoWithArgs(
                almoca.MCExploit,
                algarty.AlgoArgs(
                    sig_generator, generator_args, meta_target, max_z_ops, j_replace=1
                ),
            ),
            algabun.AlgoWithArgs(
                alave.LasVegas,
                algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
            ),
            algabun.AlgoWithArgs(
                algra.LinearRegression,
                algarty.AlgoArgs(sig_generator, generator_args, meta_target, max_z_ops),
            ),
        ]
    else:
        raise ValueError(f"Unknown selector: {selector}")

    return out_algos


algo_list: List[algo.SearchAlgo] = [
    almoca.MCExploit,
    almoca.MCExploitJ10,
    almoca.MCExploitDecoupled,
    almoca.MCExploitWeight,
    almoca.MCExploitFast,
    almoca.MCExploitNeighborWeight,
    almoca.MCOneShot,
    almoca.MCOneShotWeight,
    almoca.MCGrowShrink,
    almoca.MCDampen,
    almoca.MCPurge,
    almoca.MCOscillatorAnneal,
    almoca.MCOscillatorAnnealWeight,
    almoca.MCOscillatorAnnealLog,
    almoca.MCOscillatorAnnealLogWeight,
    alave.LasVegas,
    alave.LasVegasWeight,
    alevo.DifferentialEvolution,
    almcmc.MCExploitErgodic,
    almcmc.MCExploitAnneal,
    almcmc.MCExploitAnnealWeight,
    almcmc.BasinHopping,
    almcmc.ScipyAnneal,
    almcmc.ScipyDualAnneal,
    algra.LinearRegression,
]
