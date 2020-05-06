#!/bin/sh
#
# Download SLURM logs.
#

echo "download-logs started"
rsync -aP --include='slurm-*.out' --exclude='*' $PEREGRINE_USERNAME@peregrine.hpc.rug.nl:~/ ./logs/
echo "download-logs finished"