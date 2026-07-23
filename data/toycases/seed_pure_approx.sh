#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
export NUMBA_NUM_THREADS=4

# Définition de la variable
N=64

script_name=$0
script_full_path=$(dirname "$0")

for i in 0
do
  # Utilisation de ${N} pour construire les noms de fichiers et d'arguments
  uv run python "$script_full_path/edit_yaml.py" --file "nn_N${N}_fixed_angle_posterior-pure-mult-moment-matching.yaml" --seed "$i" --type "pure_add_approx_${N}_TSP_1p5"
  uv run python src/beetroots/simulations/astro/problems/toy_case/toy_case_nn_pure_approx.py astro_approx.yml ./data/toycases ./data/models ./old_outputs/
done