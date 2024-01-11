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
    "with_spatial_prior": {"required": True, "type": "bool"},
    "spatial_prior": {
        "required": False,
        "type": "dict",
        "schema": {
            "name": {"type": "string", "allowed": ["L2-laplacian", "L2-gradient"]},
            "use_next_nearest_neighbours": {"type": "bool"},
            "initial_regu_weights": {"type": "list"},
            "use_clustering": {"type": "bool"},
            "n_clusters": {"nullable": True, "type": "integer", "min": 2},
            "cluster_algo": {
                "nullable": True,
                "type": "string",
                "allowed": ["spectral_clustering", "kmeans"],
            },
        },
    },
}
