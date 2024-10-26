#!/bin/bash
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=64000
#SBATCH --cpus-per-task=2
#SBATCH --output=ARRARIDX.out
#SBATCH --error=ARRARIDX.err
 
cd $PWD

module purge

# module load singularity

# module load slurm
module load python/3.9.6-gc563
module load intel/17.0.4
module load gcc/11.2/openmpi/4.1.6-ez82

PARAMS_FILE="$PWD/params_APT.txt"
read -r es0 es1 esC0 esC1 p_m L <<< $(sed -n "ARRARIDXp" $PARAMS_FILE)

echo $es0 $es1 $esC0 $esC1 $p_m $L
srun --mpi=pmi2 python -m mpi4py.futures run_APT_C_m.py -es $es0 $es1 -es_C $esC0 $esC1 -p_m $p_m $p_m 1 -p_f 1 1 1 -L $L

