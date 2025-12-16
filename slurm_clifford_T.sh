#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=Clifford_T
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=hnpanboa@gmail.com
#SBATCH --requeue
#SBATCH --output=slurm_out/Clifford_T_%A_%a.out
#SBATCH --array=1

# Set number of parallel jobs based on allocated CPUs
N_JOBS="$SLURM_CPUS_PER_TASK"

# Thread control for numerical libraries
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Setup Python environment
source ~/.bashrc
pyenv shell miniforge3-25.1.1-2
PYTHON_PATH="$HOME/.pyenv/versions/miniforge3-25.1.1-2/bin/python"
echo "Python path: $(which python)"

# Read parameters for this job array index using REAL_TASK_ID (set by job_manager.py)
read -r PARAMS <<< "$(sed -n "${REAL_TASK_ID}p" "$PARAMS_FILE")"

echo "Job array index: $SLURM_ARRAY_TASK_ID"
echo "Real task ID: $REAL_TASK_ID"
echo "Parameters: $PARAMS"
echo "Using $N_JOBS parallel jobs"

# Run the simulation with full parameter string
srun $PYTHON_PATH $PWD/run_Clifford_T.py \
    $PARAMS \
    --n_jobs $N_JOBS

echo "Completed job array index: $SLURM_ARRAY_TASK_ID (task $REAL_TASK_ID)"
