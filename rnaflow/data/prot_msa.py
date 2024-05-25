import os
import subprocess
import shutil
from multiprocessing import Pool

# Function to process each subfolder
def process_subfolder(subfolder):
    subfolder_path = os.path.join(main_folder, subfolder)

    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Path to the prot.fa file
        prot_fa_path = os.path.join(subfolder_path, "prot.fa")

        # Check if prot.fa file exists in the subfolder
        if os.path.exists(prot_fa_path):
            # Path to the destination folder
            destination_folder = os.path.join(main_folder, subfolder)

            # Path to the destination file
            destination_file_path = os.path.join(destination_folder, "prot.a3m")

            # Check if the destination file has more than 5 lines
            with open(destination_file_path, 'r') as dest_file:
                if sum(1 for line in dest_file) <= 5:
                    # Run the bash script
                    subprocess.run([rose_script, f"{output_folder}_{subfolder}", prot_fa_path])

                    # Path to the output file
                    output_file_path = os.path.join(f"{output_folder}_{subfolder}", "prot.msa0.a3m")

                    # Copy the output file to the destination folder
                    shutil.copy(output_file_path, destination_file_path)
                    shutil.rmtree(f"{output_folder}_{subfolder}")
                    print(f"Processed {subfolder}.")
                else:
                    print(f"Skipped {subfolder} as the destination file already has more than 5 lines.")

# Path to the main folder
main_folder = "rnaflow/data/rf_data"

# Path to the RoseTTAFold2NA script
rose_script = "RoseTTAFold2NA/run_RF2NA.sh"

# Path to the output folder
output_folder = "RoseTTAFold2NA/example/rna_pred"

# List of subfolders
subfolders = os.listdir(main_folder)

# Number of processes (change this based on your system)
num_processes = 64

# Use multiprocessing Pool to parallelize execution
with Pool(num_processes) as pool:
    pool.map(process_subfolder, subfolders)

print("Task completed.")