#!/bin/bash
echo "Name,Logpath,State,Reserved walltime,Used walltime,Used CPU time,Nodes,Max mem used,Max disk write,Max disk read,Start,End" >> $1/all_logs.csv
for filename in $1/*.out; do
    echo $filename
    name=$(grep $filename -e       'Name                :' | cut -d":" -f 2- )
    logpath=$filename
    state=$(grep $filename -e      'State               :' | cut -d":" -f 2- )
    r_walltime=$(grep $filename -e 'Reserved walltime   :' | cut -d":" -f 2- )
    u_walltime=$(grep $filename -e 'Used walltime       :' | cut -d":" -f 2- )
    cpu=$(grep $filename -e        'Used CPU time       :' | cut -d":" -f 2- )
    nodes=$(grep $filename -e      'Nodes               :' | cut -d":" -f 2- )
    max_mem=$(grep $filename -e    'Max Mem used        :' | cut -d":" -f 2- )
    max_write=$(grep $filename -e  'Max Disk Write      :' | cut -d":" -f 2- )
    max_read=$(grep $filename -e   'Max Disk Read       :' | cut -d":" -f 2- )
    start=$(grep $filename -e      'Start               :' | cut -d":" -f 2- )
    end=$(grep $filename -e        'End                 :' | cut -d":" -f 2- )
    echo "\"$name\",\"$logpath\",\"$state\",\"$r_walltime\",\"$u_walltime\",\"$cpu\",\"$nodes\",\"$max_mem\",\"$max_write\",\"$max_read\",\"$start\",\"$end\"" >> $1/all_logs.csv
done
