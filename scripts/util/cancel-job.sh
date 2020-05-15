#!/bin/sh
#
# Logs into Peregrine and cancels one or multiple jobs.
# e.g. scripts/cancel-job 11321699 11386210
#

echo "cancel-job started"
ssh $PEREGRINE_USERNAME@peregrine.hpc.rug.nl "scancel $@"
echo "cancel-job finished"