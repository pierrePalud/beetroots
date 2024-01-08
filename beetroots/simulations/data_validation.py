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
    }
}
