#!/bin/bash -l
#COBALT -n 1
#COBALT -t 60
#COBALT -q full-node
#COBALT -A datascience
#COBALT --attrs filesystems=home,theta-fs0

echo loading module $1
module load $1
conda activate

module list

echo python = $(which python)

NODES=`cat $COBALT_NODEFILE | wc -l`
GPUS_PER_NODE=8
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  GPUS_PER_NODE=$GPUS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=16

mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np ${RANKS} -npernode ${GPUS_PER_NODE} --hostfile ${COBALT_NODEFILE} python /home/parton/git/randles_esp/run_training_v2.py
