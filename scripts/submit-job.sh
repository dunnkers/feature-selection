#!/bin/sh
#
# Logs into Peregrine and executes a job.
#

./scripts/upload-files.sh

echo "submit-job started"
ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "sbatch ${1}"
echo "submit-job finished"