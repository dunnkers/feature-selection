#!/bin/sh
#
# Download SLURM logs and csv output.
#

echo "download-output started"
rsync -aP --include='logs/***' --include='results/***' --exclude='*' $PEREGRINE_USERNAME@peregrine.hpc.rug.nl:~/ ./
echo "download-output finished"