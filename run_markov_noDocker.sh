#!/bin/bash
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --time=13:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=64000
#SBATCH --cpus-per-task=1
#SBATCH --output=ARRARIDX.out
#SBATCH --error=ARRARIDX.err

cd $PWD

module purge

module load python/3.9.6-gc563
module load intel/17.0.4
module load openmpi/4.1.6

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PARAMS_FILE="$PWD/params_markov.txt"
read -r p L <<< $(sed -n "ARRARIDXp" $PARAMS_FILE)

echo $p $L

srun --mpi=pmi2 python3 -m mpi4py.futures run_markov.py -p $p $p 1 -L $L
