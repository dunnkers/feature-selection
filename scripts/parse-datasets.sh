#!/bin/sh
# 
# Submits all directory paths from local datasets located in `./data/` to a 
# local .csv file. Only saves datasets that are split up in 30 different 
# files. (e.g. datasets from Urbanowicz et al, 2018) 
# 

rm to-submit.csv
echo 'dataset_path' >> to-submit.csv
for i in `find ./data -type d `; do
    if [ -d $i ]; then
        j=`ls -1 $i | wc -l`;
        if [ $j -eq  30 ]; then
            echo $i >> to-submit.csv
        fi;
    fi;
done