param_sweep_schedule = {
    "vo2_r1": {
        "changed_component": "R1",
        "dependent_component": "v(\"/A\")",
        
        "start": "5k",
        "stop": "150k",
        "step": "1k",

        "time_step": "5e-9",
        "time_stop": "10u",
        "time_start": "0",
    },
}

# parameters to generate bird sounds
bird_params = {
        "magpie": {
        "num_osc": 1,

        "v_in": 14,

        "r_last": 30e3,

        "r_min": 30e3,
        "r_max": 60e3,
        "r_dist": "uniform",

        "c_min": 300e-12,
        "c_max": 300e-12,
        "c_dist": "uniform",

        "r_out_min": 1,
        "r_out_max": 1,
        "r_out_dist": "uniform",

        "time_step": 5e-9,
        "time_stop": "10u",
        "time_start": "0",

        "dependent_component": "v(\"/sum\")",

        "file_name": "magpie",
    }
}