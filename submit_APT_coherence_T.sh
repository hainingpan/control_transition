#!/bin/bash
#PBS -A ONRDC54450755
#PBS -l walltime=4:00:00
#PBS -q standard
#PBS -l select=1:ncpus=192:mpiprocs=1
#PBS -N APT_coherence_T
#PBS -m abe
#PBS -M hnpanboa@gmail.com
#PBS -r y
#PBS -J 1-243

# Set number of parallel jobs
N_JOBS=192

# Setup Python environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"
pyenv shell miniforge3-25.1.1-2

# Read parameters for this job array index
PARAMS_FILE="$HOME/control_transition/params_APT_coherence_T_2.txt"
PARAMS=$(sed -n "${PBS_ARRAY_INDEX}p" $PARAMS_FILE)

echo "Job array index: $PBS_ARRAY_INDEX"
echo "Parameters: $PARAMS"

# Run the simulation with full parameter string
python $HOME/control_transition/run_APT_coherence_T.py \
    $PARAMS \
    --p_f 0 0 -1 \
    --n_jobs $N_JOBS

echo "Completed job array index: $PBS_ARRAY_INDEX"
