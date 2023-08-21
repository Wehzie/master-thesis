"""Parameters for running SPICE simulations with the SPICE signal generator."""

# parameters to generate bird sounds
bird_params = {
    "magpie_single_oscillator": {
        "trials": 1,
        "n_osc": 1,
        "v_in": 14,
        "r_last": 0,
        "r_control": 1e6,
        "r_min": 30e3,
        "r_max": 70e3,
        "r_dist": "uniform",
        "c_min": 300e-12,
        "c_max": 300e-12,
        "c_dist": "uniform",
        "time_step": 5e-9,
        "time_stop": 1e-5,
        "time_start": 0,
        "dependent_component": "v(osc1)",
    },
    "magpie_sum": {
        "trials": 1,
        "n_osc": 1,
        "v_in": 4,
        "r_last": 1,  # only with sum architecture
        "r_control": 1e6,
        "r_min": 8e3,
        "r_max": 8e3,
        "r_dist": "uniform",
        "c_min": 40e-12,
        "c_max": 100e-12,
        "c_dist": "uniform",
        "time_step": 5e-9,
        "time_stop": "10e-6",
        "time_start": "0",
        "dependent_component": "v(sum)",
    },
    "magpie_tree": {
        "trials": 1,
        "branching": 2,  # branching factor
        "height": 5,  # tree height
        "v_in": 14,
        "r_tree": 0,  # only with tree architecture
        "r_control": 1e6,
        "r_min": 30e3,
        "r_max": 70e3,
        "r_dist": "uniform",
        "c_min": 300e-12,
        "c_max": 300e-12,
        "c_dist": "uniform",
        "time_step": 5e-9,
        "time_stop": "1e-5",
        "time_start": "0",
        "dependent_component": "v(wire0)",
    },
}
