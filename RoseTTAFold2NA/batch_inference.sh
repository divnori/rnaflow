#!/bin/bash

folder_path="aptamer_data/GRK2/"
all_files=("${folder_path}"/*)


# # Iterate over each file in the folder
for file in "${all_files[@]}"
do

    filename=$(basename "${file}")   # Extract the filename from the file path
    name="${filename%.*}"           # Remove the extension to get the name
    extension="${filename##*.}"     # Extract the extension
    
    ./run_RF2NA.sh ${file}/rna_pred R:${file}/RNA.fa

done

# ./run_RF2NA.sh ${file}/rna_pred P:${file}/protein.fa R:${file}/RNA.fa use_rna_msa:0 device:1
