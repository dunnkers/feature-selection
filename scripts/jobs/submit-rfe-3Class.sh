#!/bin/sh

for f in data/3Class_Datasets_Loc_2/*.txt.gz;
    do echo "Submitting $f.." && \
    scripts/submit-job.sh fs-methods/rfe.py $f;
done