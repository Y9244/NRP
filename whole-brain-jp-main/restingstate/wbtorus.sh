#!/bin/bash

#PJM -L "node=48"
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=small-s1"
#PJM -L "elapse=01:00:00"
#PJM -g "hp200139"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM --mpi "max-proc-per-node=4"
#PJM -S

. /vol0004/apps/oss/spack/share/spack/setup-env.sh

spack load gsl
spack load /c4nk455 # python@3.8.12%fj@4.7.0
spack load /vzyz2hf # py-matplotlib@3.4.3
spack load /o3x5jab # py-scipy@1.6.1

export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=12

source /home/hp200139/data/nest220gcc/bin/nest_vars.sh

mpiexec python -u stim_all_model.py
