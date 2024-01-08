#!/bin/bash
#SBATCH --job-name=python_inversions
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=7100
#SBATCH --partition=long
#SBATCH --mem=4gb
#SBATCH --error=run_%A_%a.err
#SBATCH --mail-user=pierre.palud@obspm.fr
#SBATCH --mail-type=ALL
export OMP_NUM_THREADS=1

# To be run with `sbatch job_run_one_file.slurm <input_file.yaml> <path to input_file>`
# example: sbatch job_run_one_file.slurm input_file.yaml .

# create some variables to facilitate files transfers
SCRATCH=/scratch/$USER/run_${1::-5}
DATA=/poubelle/$USER/test_regularization_params/run_${1::-5}


# transfer files to sratch to run the PDR code there
mkdir $SCRATCH
cp -r . $SCRATCH
cd $SCRATCH
mkdir outputs

echo $SLURM_JOB_ID
echo $SCRATCH
echo $DATA

CURRENT_PATH=$(pwd)
echo $CURRENT_PATH

poetry run python beetroots/simulations/astro/toy_case/toy_case_nn_direct_posterior.py ${2}/${1}

# delete resulting hdf5 files, as they are usually heavy
find ./outputs -name "*.hdf5" -type f -delete

# transfer results to /poubelle/$USER
mkdir $DATA
tar -cvzf $DATA/result-${1::-5}.tar.gz ./outputs

# delete scratch
rm -rf $SCRATCH

exit 0
