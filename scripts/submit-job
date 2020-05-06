#!/bin/sh
#
# Logs into Peregrine and executes a job.
#

echo "submit-job started"
ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "sbatch python_batch.sh ${1} ${2}"
echo "submit-job finished"