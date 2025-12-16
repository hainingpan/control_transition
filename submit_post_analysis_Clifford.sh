#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --job-name=PostAnalysis_Clifford
#SBATCH --requeue
#SBATCH --output=slurm_out/PostAnalysis_Clifford_%A_%a.out
#SBATCH --array=0-5

# Thread control for numerical libraries
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Setup Python environment
source ~/.bashrc
pyenv shell miniforge3-25.1.1-2
PYTHON_PATH="$HOME/.pyenv/versions/miniforge3-25.1.1-2/bin/python"

# Define parameter arrays
L_VALUES=(256)
PM_VALUES=(0.5 0.55 0.6 0.65 0.7 0.75)

# Calculate indices from SLURM_ARRAY_TASK_ID
# Total combinations = 5 * 6 = 30
# Index logic: L varies slower (outer loop equivalent), or p_m varies slower?
# Let's vary p_m faster (inner loop) as is common.
# ID = L_Index * 6 + PM_Index
L_IDX=$((SLURM_ARRAY_TASK_ID / 6))
PM_IDX=$((SLURM_ARRAY_TASK_ID % 6))

L=${L_VALUES[$L_IDX]}
PM=${PM_VALUES[$PM_IDX]}

echo "=== Post Analysis for L=$L, p_m=$PM ==="
echo "Job array index: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

# Run the analysis script
$PYTHON_PATH $PWD/post_analysis_Clifford_fluct_T_all.py \
    --L $L \
    --p_m $PM \
    --alpha 0.5 \
    --ob OP

echo "End time: $(date)"
echo "=== Completed ==="
