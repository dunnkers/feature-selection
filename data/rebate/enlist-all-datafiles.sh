# for k in {1..30}; do ### all datasets ordered by replica_no
    for i in `find . -type d `; do
        if [ -d $i ]; then
            j=`ls -1 $i | wc -l`;
            if [ $j -eq  30 ]; then
                cnt=0
                for file in `ls $i`; do
                    ((cnt+=1))
                    ### all datasets, ordered by dataset
                    if [ $cnt -le 24 ]; then
                        echo ,,rebate,$i,$i/$file,$cnt,24 # also replicas
                    fi;

                    ### all datasets ordered by replica_no
                    # if [ $cnt -eq $k ]; then
                    #     # echo $i/$file # file only
                    #     echo rebate,$i/$file,$k,30 # also replicas
                    # fi;
                done
            fi;
        fi;
    done
    # break 1; # uncomment to show only 1 file
# done