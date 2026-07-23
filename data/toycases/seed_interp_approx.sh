#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

script_name=$0
script_full_path=$(dirname "$0")

for i in 0
do
  uv run python $script_full_path/edit_yaml.py nn_N64_fixed_angle_posterior-approx.yaml $i approx64_beta_2p0
  uv run python src/beetroots/simulations/astro/problems/toy_case/toy_case_nn_approx.py astro_approx.yml ./data/toycases ./data/models ./old_outputs/
done
