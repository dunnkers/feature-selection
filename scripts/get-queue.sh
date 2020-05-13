#!/bin/sh
#
# Retrieves your job queue from SLURM.
#

echo "get-queue started"
ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "squeue -u $PEREGRINE_USERNAME"
echo "get-queue finished"