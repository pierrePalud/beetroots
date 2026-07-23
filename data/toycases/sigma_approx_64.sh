#!/bin/bash
set -euo pipefail

echo "Bash version ${BASH_VERSION}..."
export NUMBA_NUM_THREADS=4

script_full_path=$(dirname "$0")
base_yaml="nn_N64_fixed_angle_posterior-approx.yaml"
seed=0

# sigma_m -> matching normalization fit file
declare -A BEST_PARAMS_BY_SIGMA=(
  ["1.1"]="best_params_exppoly_a10_m1p1.csv"
  ["1.5"]="best_params_exppoly_a10_m1p5.csv"
  ["2.0"]="best_params_exppoly_a10_m2p0.csv"
)

pids=()

for s in 1.1 1.5 2.0
 do
  tag=${s//./p}
  yaml_out="astro_approx_N64_sigma_${tag}.yml"
  best_params_file="${BEST_PARAMS_BY_SIGMA[$s]}"

  echo "Preparing YAML for sigma_m=${s}, best_params=${best_params_file}, rng_seed=${seed}"

  uv run python - <<PY
import os
import yaml

base_path = os.path.join("${script_full_path}", "${base_yaml}")
out_path = os.path.join("${script_full_path}", "${yaml_out}")

with open(base_path, "r", encoding="utf-8") as f:
    y = yaml.safe_load(f)

y["simu_init"]["simu_name"] = "astro_approx_N64_fixed_angle_sigma_${tag}_seed${seed}"
y["sigma_m_float_linscale"] = float("${s}")
y["component_distributions"]["approx-likelihood"]["params"]["mixing_model_params_filename"] = ["${best_params_file}"]
y["run_params"]["map"]["rng_seed"] = int(${seed})
y["run_params"]["mcmc"]["rng_seed"] = int(${seed})

with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(y, f, default_flow_style=False, sort_keys=False)
PY

  echo "Launching simulation for sigma_m=${s} in background"
  uv run python src/beetroots/simulations/astro/problems/toy_case/toy_case_nn_approx.py \
    "${yaml_out}" ./data/toycases ./data/models ./old_outputs/ &

  pids+=("$!")
 done

echo "Launched ${#pids[@]} simulations in parallel. Waiting for completion..."
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All simulations completed."
