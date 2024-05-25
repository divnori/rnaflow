#!/bin/bash

# make the script stop when error (non-true exit code) occurs
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

SCRIPT=`realpath -s $0`
#export PIPEDIR=`dirname $SCRIPT`
export PIPEDIR="/home/huangtin/RoseTTAFold2NA"
HHDB="$PIPEDIR/pdb100_2021Mar03/pdb100_2021Mar03"

CPU="16"  # number of CPUs to use
MEM="64" # max memory (in GB)

conda activate RF2NA

rf_data_drectory="rnaflow/data/rf_data"
directories=$(find "$rf_data_drectory" -type d)

for dir in $directories; do
    echo "Processing directory: $dir"

    # Check if dir does not end with "data"
    if [[ ! "$dir" =~ data$ ]]; then
        echo "Running hhsearch"
        HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $HHDB"
        echo " -> Running command: $HH -i $dir/prot.a3m -o $WDIR/$tag.hhr -atab $WDIR/$tag.atab -v 0"
        chmod +r "$dir/prot.a3m"
        $HH -i "$dir/prot.a3m" -o "$dir/prot.hhr" -atab "$dir/prot.atab" -v 0
    else
        echo "Skipping hhsearch for directory ending with 'data'"
    fi

done