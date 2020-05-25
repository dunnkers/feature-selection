#!/bin/bash
echo "Name,Logpath,State,Reserved walltime,Used walltime,Used CPU time,Nodes,Max mem used,Max disk write,Max disk read,Start,End"
for filename in "${@:1:$#-1}"; do
    name=$(grep $filename -e       'Name                :' | cut -d":" -f 2- | awk '{$1=$1};1')
    logpath=$filename
    state=$(grep $filename -e      'State               :' | cut -d":" -f 2- | awk '{$1=$1};1')
    r_walltime=$(grep $filename -e 'Reserved walltime   :' | cut -d":" -f 2- | awk '{$1=$1};1')
    u_walltime=$(grep $filename -e 'Used walltime       :' | cut -d":" -f 2- | awk '{$1=$1};1')
    cpu=$(grep $filename -e        'Used CPU time       :' | cut -d":" -f 2- | awk '{$1=$1};1')
    nodes=$(grep $filename -e      'Nodes               :' | cut -d":" -f 2- | awk '{$1=$1};1')
    max_mem=$(grep $filename -e    'Max Mem used        :' | cut -d":" -f 2- | awk '{$1=$1};1')
    max_write=$(grep $filename -e  'Max Disk Write      :' | cut -d":" -f 2- | awk '{$1=$1};1')
    max_read=$(grep $filename -e   'Max Disk Read       :' | cut -d":" -f 2- | awk '{$1=$1};1')
    start=$(grep $filename -e      'Start               :' | cut -d":" -f 2- | awk '{$1=$1};1')
    end=$(grep $filename -e        'End                 :' | cut -d":" -f 2- | awk '{$1=$1};1')
    echo "\"$name\",\"$logpath\",\"$state\",\"$r_walltime\",\"$u_walltime\",\"$cpu\",\"$nodes\",\"$max_mem\",\"$max_write\",\"$max_read\",\"$start\",\"$end\""
done
