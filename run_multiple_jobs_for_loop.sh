#!/bin/bash

CURRENT_ROOT=$(pwd);
# echo $CURRENT_ROOT

# get list of input files
prefix=./data/toycases/run_test_regularization_effect;
cd $prefix;
FILES=($(ls -1));
echo "${FILES[@]}"

cd $CURRENT_ROOT;

for file in "${FILES[@]}"; do
    echo "run_test_regularization_effect/${file}";
    sbatch job_run_one_file.slurm $file "run_test_regularization_effect";
done
