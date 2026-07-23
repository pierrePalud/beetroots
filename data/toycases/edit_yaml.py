#!/usr/bin/python

import argparse
import os

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit YAML file")
    parser.add_argument("--file", type=str, help="Path to the YAML file")
    parser.add_argument("--seed", type=int, help="Seed value", default=0)
    parser.add_argument("--type", type=str, help="Type of the simulation")
    parser.add_argument(
        "--sigma-m",
        type=float,
        default=-1.0,
        help="Sigma_m value (optional, default: unchanged)",
    )
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.file)) as f:
        y = yaml.safe_load(f)
        y["simu_init"]["simu_name"] = f"astro_{args.type}_seed{args.seed}"
        y["run_params"]["mcmc"]["rng_seed"] = args.seed
        y["run_params"]["map"]["rng_seed"] = args.seed
        if args.sigma_m != -1.0:
            y["sigma_m_float_linscale"] = args.sigma_m
    with open(
        os.path.join(os.path.dirname(__file__), "astro_approx.yml"), "w"
    ) as ostream:
        yaml.dump(y, ostream, default_flow_style=False, sort_keys=False)
