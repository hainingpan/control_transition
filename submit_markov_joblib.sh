#!/bin/bash
#PBS -A ONRDC54450755
#PBS -l walltime=4:02:00
#PBS -q standard
#PBS -l select=1:ncpus=192:mpiprocs=1
#PBS -N markov
#PBS -m abe
#PBS -M hnpanboa@gmail.com
#PBS -r y
#PBS -J 1-8
cd ~/control_transition

# Set the number of parallel jobs to use all available CPUs
N_JOBS=192

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"
pyenv shell miniforge3-25.1.1-2

# No need to load openmpi for joblib

# Get the parameters for this job array index
read -r p L <<< $(sed -n "${PBS_ARRAY_INDEX}p" params.txt)

echo "Job array index: $PBS_ARRAY_INDEX"
echo "Running with p=$p, L=$L"

# Run with joblib (no mpirun needed)
python run_markov_joblib.py -p $p $p 1 -L $L --n_jobs $N_JOBS

echo "Completed p=$p, L=$L"




