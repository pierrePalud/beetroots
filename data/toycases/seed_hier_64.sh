#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

script_name=$0
script_full_path=$(dirname "$0")

for i in 2 3 4
do
  echo "Running simulation with seed $i"
  uv run python $script_full_path/edit_yaml.py nn_N64_fixed_angle_hierarchical.yaml $i hier_N64_fixed_angle
  uv run python src/beetroots/simulations/astro/problems/toy_case/toy_case_nn_hierarchical.py astro_approx.yml ./data/toycases ./data/models ./old_outputs/
done
