#!/bin/bash
#SBATCH --account=ONRDC54450755
#SBATCH --time=4:00:00
#SBATCH -q background
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=Clifford_T
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=hnpanboa@gmail.com
#SBATCH --requeue
#SBATCH --output=slurm_out/Clifford_T_%A_%a.out
#SBATCH --array=1-999

cd $HOME/control_transition

# Set number of parallel jobs (bound to cpus-per-task)
N_JOBS=${SLURM_CPUS_PER_TASK:-128}

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
PARAMS_FILE="$HOME/control_transition/params_clifford_T_2.txt"

# Calculate actual parameter line using OFFSET (default 0)
OFFSET=${OFFSET:-0}
REAL_TASK_ID=$((OFFSET + SLURM_ARRAY_TASK_ID))

# Read parameters for this job array index
read -r PARAMS <<< "$(sed -n "${REAL_TASK_ID}p" "$PARAMS_FILE")"

echo "Job array index: $SLURM_ARRAY_TASK_ID"
echo "Offset: $OFFSET"
echo "Real task ID (parameter line): $REAL_TASK_ID"
echo "Parameters: $PARAMS"
echo "Using $N_JOBS parallel jobs"

# Run the simulation with full parameter string
$PYTHON_PATH $PWD/run_Clifford_T.py \
    $PARAMS \
    --n_jobs $N_JOBS

echo "Completed job array index: $SLURM_ARRAY_TASK_ID (task $REAL_TASK_ID)"
