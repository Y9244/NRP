#!/bin/bash

#PJM -L "node=64"
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=small"
#PJM -L "elapse=01:00:00"
#PJM -S

. /vol0004/apps/oss/spack/share/spack/setup-env.sh

spack load gsl
spack load /c4nk455 # python@3.8.12%fj@4.7.0
spack load /vzyz2hf # py-matplotlib@3.4.3
spack load /o3x5jab # py-scipy@1.6.1

export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH

source /home/hp200139/data/nest220gcc/bin/nest_vars.sh

mpiexec python stim_all_model.py
