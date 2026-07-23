#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

script_name=$0
script_full_path=$(dirname "$0")

for i in {1..5..1}
do
  echo "Running simulation with seed $i"
  python $script_full_path/edit_yaml.py nn_N10_fixed_angle_hierarchical.yaml $i hier_N10_EUSIPCO26_fixed_angle
  python beetroots/simulations/astro/problems/toy_case/toy_case_nn_hierarchical.py astro_approx.yml ./data/toycases ./data/models .
done
