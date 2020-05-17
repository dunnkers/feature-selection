for k in {1..30}; do
    for i in `find . -type d `; do
        if [ -d $i ]; then
            j=`ls -1 $i | wc -l`;
            if [ $j -eq  30 ]; then
                cnt=0
                for file in `ls $i`; do
                    ((cnt+=1))
                    if [ $cnt -eq $k ]; then
                        echo $i/$file
                    fi;
                    # break 1; # uncomment to show only 1 file
                done
            fi;
        fi;
    done
done