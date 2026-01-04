#!/bin/bash
#PBS -A ONRDC54450755
#PBS -l walltime=0:59:00
#PBS -q background
#PBS -l select=1:ncpus=192:mpiprocs=1
#PBS -N PostAnalysis_Clifford
#PBS -m abe
#PBS -M hnpanboa@gmail.com
#PBS -r y
#PBS -J 0-23

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
# L_VALUES=(16 32 64)
L_VALUES=(128 )
# L_VALUES=(256 )
# PM_VALUES=(0.5  0.55 0.6  0.64 0.65 0.66 0.67 0.68 0.69 0.7  0.71 0.72 0.73 0.74  0.75)
# PM_VALUES=(0.662 0.664 0.666 0.668 0.672 0.674 0.676 0.678)
PM_VALUES=(0.5 0.55 0.6 0.64 0.65 0.66 0.662 0.664 0.666 0.668 0.67 0.672 0.674 0.676 0.678 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75)

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
$PYTHON_PATH $PWD/post_analysis_Clifford_fluct_T_all.py \
    --L $L \
    --p_m $PM \
    --alpha 0.5 \
    --ob OP

echo "End time: $(date)"
echo "=== Completed ==="
