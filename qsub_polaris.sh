#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l walltime=00:60:00
#PBS -q debug
#PBS -A datascience
#PBS -o logs/$PBS_JOBID.output
#PBS -e logs/$PBS_JOBID.error

# job doesn't start in work directory
cd $PBS_O_WORKDIR

echo loading module
module load conda/2022-07-19
conda activate

#export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH

module list

echo python = $(which python)

NODES=`cat $PBS_NODEFILE | wc -l`
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  GPUS_PER_NODE=$GPUS_PER_NODE  RANKS=$RANKS
# 64 CPU cores total
export OMP_NUM_THREADS=16
echo OMP_NUM_THREADS=$OMP_NUM_THREADS

mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=$OMP_NUM_THREADS --cpu-bind depth --env OMP_NUM_THREADS=$OMP_NUM_THREADS -env OMP_PLACES=threads \
   python  -c "import mpi4py.MPI as mpi;import socket;print('host: ',socket.gethostname(), 'rank: ',mpi.COMM_WORLD.Get_rank(),' size: ',mpi.COMM_WORLD.Get_size())"

# HOROVOD_TIMELINE=filename
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=$OMP_NUM_THREADS --cpu-bind depth --env OMP_NUM_THREADS=$OMP_NUM_THREADS -env OMP_PLACES=threads \
   python /home/parton/git/randles_esp/run_training_v4.py
