#!/bin/bash
#PBS -A ONRDC54450755
#PBS -l walltime=0:30:00
#PBS -q standard
#PBS -l select=1:ncpus=192:mpiprocs=1
#PBS -N PostAnalysis_Clifford
#PBS -m abe
#PBS -M hnpanboa@gmail.com
#PBS -r y
#PBS -J 0-20

cd $HOME/control_transition

# Thread control for numerical libraries
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Setup Python environment
source ~/.bashrc
pyenv shell miniforge3-25.1.1-2
PYTHON_PATH="$HOME/.pyenv/versions/miniforge3-25.1.1-2/bin/python"

# Define parameter arrays
L_VALUES=(12 14 16)
# L_VALUES=(12 14 16 18 20)
# L_VALUES=(128 )
# L_VALUES=(256 )
# PM_VALUES=(0.085 0.087 0.089 0.09 0.091 0.093 0.095)
PM_VALUES=(0.05 0.06 0.07 0.08 0.10 0.11 0.12)

# Calculate indices from PBS_ARRAY_INDEX
# Index logic: L varies slower, p_m varies faster (inner loop)
# ID = L_Index * NUM_PM + PM_Index
NUM_PM=${#PM_VALUES[@]}
L_IDX=$((PBS_ARRAY_INDEX / NUM_PM))
PM_IDX=$((PBS_ARRAY_INDEX % NUM_PM))

L=${L_VALUES[$L_IDX]}
PM=${PM_VALUES[$PM_IDX]}

echo "=== Post Analysis for L=$L, p_m=$PM ==="
echo "Job array index: $PBS_ARRAY_INDEX"
echo "Start time: $(date)"

# Run the analysis script
$PYTHON_PATH $PWD/post_analysis_APT_fluct_T_all.py \
    --L $L \
    --p_m $PM \
    --ob OP --threshold 1e-8

echo "End time: $(date)"
echo "=== Completed ==="