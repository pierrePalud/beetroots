r"""contains the validation schema that ensures that the ``.yaml`` input file, that depicts the inversion configuration, has a correct structure.
"""

schema = {
    "simu_init": {
        "required": True,
        "type": "dict",
        "schema": {
            "simu_name": {"required": True, "type": "string"},
            "cloud_name": {"required": True, "type": "string"},
            "max_workers": {"required": True, "type": "number", "min": 1, "max": 20},
            # "params_names": {},
        },
    },
    #
    "to_run_optim_map": {"required": True, "type": "boolean"},
    "to_run_mcmc": {"required": True, "type": "boolean"},
    #
    # spatial prior
    "with_spatial_prior": {"required": True, "type": "boolean"},
    "spatial_prior": {
        "required": False,
        "type": "dict",
        "schema": {
            "name": {"type": "string", "allowed": ["L2-laplacian", "L2-gradient"]},
            "use_next_nearest_neighbors": {"type": "boolean"},
            "initial_regu_weights": {"type": "list"},
        },
    },
    # sampling params
}
r"""validation schema that ensures that the ``.yaml`` input file, that depicts the inversion configuration, has a correct structure"""
