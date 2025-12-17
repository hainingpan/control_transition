#!/bin/bash
#PBS -A ONRDC54450755
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l select=1:ncpus=192:mpiprocs=1
#PBS -N Clifford_T
#PBS -m abe
#PBS -M hnpanboa@gmail.com
#PBS -r y
#PBS -J 10001-11000

cd $HOME/control_transition

# Set number of parallel jobs based on allocated CPUs
N_JOBS=125

# Thread control for numerical libraries
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Setup Python environment
source ~/.bashrc
pyenv shell miniforge3-25.1.1-2
PYTHON_PATH="$HOME/.pyenv/versions/miniforge3-25.1.1-2/bin/python"
echo "Python path: $(which python)"

# Parameters file
PARAMS_FILE="$HOME/control_transition/params_clifford_T.txt"

# Read parameters for this job array index
read -r PARAMS <<< "$(sed -n "${PBS_ARRAY_INDEX}p" "$PARAMS_FILE")"

echo "Job array index: $PBS_ARRAY_INDEX"
echo "Parameters: $PARAMS"
echo "Using $N_JOBS parallel jobs"

# Run the simulation with full parameter string
$PYTHON_PATH $PWD/run_Clifford_T.py \
    $PARAMS \
    --n_jobs $N_JOBS

echo "Completed job array index: $PBS_ARRAY_INDEX"
