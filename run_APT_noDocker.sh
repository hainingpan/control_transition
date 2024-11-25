#!/bin/bash
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=21
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

PARAMS_FILE="$PWD/params_APT.txt"
read -r  esC0 esC1 es0 es1  p_m L <<< $(sed -n "ARRARIDXp" $PARAMS_FILE)

echo $esC0 $esC1 $es0 $es1 $p_m $L
# srun --mpi=pmi2 python3 -m mpi4py.futures run_APT_C_m.py -es $es0 $es1 -es_C $esC0 $esC1 -p_m $p_m $p_m 1 -p_f 1 1 1 -L $L
srun --mpi=pmi2 python3 -m mpi4py.futures run_APT_C_m_final.py -es_C $esC0 $esC1 -es $es0 $es1  -p_m $p_m $p_m 1 -p_f 1 1 1 -L $L
