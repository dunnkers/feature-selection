for i in `find . -type d `; do
    if [ -d $i ]; then
        j=`ls -1 $i | wc -l`;
        if [ $j -eq  30 ]; then
            for file in `ls $i`; do
                echo $i/$file
                # break 1; # uncomment to show only 1 file
            done
        fi;
    fi;
done
