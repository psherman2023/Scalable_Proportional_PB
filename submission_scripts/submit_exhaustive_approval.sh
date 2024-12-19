#!/bin/bash

# Folder containing files
FILES_DIR="/data/coml-humanchess/univ5678/PB_Data/PB/2024-09-26_20-45-53_pabulib_approval_exhaustive_2"

# FILES_DIR="/data/coml-humanchess/univ5678/PB_Data/Test"

# SLURM options
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:5:00
#SBATCH --partition short
#SBATCH --output=out.log

# Load necessary modules (if any)
# module load exact_method_of_equal_shares_with_completion_approval
module load Anaconda3 
# Check if pabutools is already installed
if ! python3 -c "import pabutools" &> /dev/null; then
    echo "PABUTools is not installed. Installing it now..."
    
    # Load Python or Anaconda module if needed (uncomment if needed)
    # module load python/3.8  # Adjust the Python version if needed
    # module load anaconda3    # If using Anaconda

    # Install PABUTools using pip
    pip install --user pabutools  # Use --user to install in user space on clusters
fi



# Check if pabutools is already installed
if ! python3 -c "import pandas" &> /dev/null; then
    echo "pandas is not installed. Installing it now..."
    
    # Load Python or Anaconda module if needed (uncomment if needed)
    # module load python/3.8  # Adjust the Python version if needed
    # module load anaconda3    # If using Anaconda

    # Install PABUTools using pip
    pip install --user pandas  # Use --user to install in user space on clusters
fi

# module load pabutools
# module load pandas


# Path to the script with your custom function
CUSTOM_FUNCTION_PATH="/data/coml-humanchess/univ5678/PB_scripts/run_approval_equal_shares_exhaustive.py"


# Loop through all files in the directory
for file in "$FILES_DIR"/*
do
    # Extract the filename without path
    filename=$(basename -- "$file")
    
    # Submit a job for each file
   sbatch --partition=long --ntasks=1 --cpus-per-task=1 --job-name="$filename" --output="result_logs/results_exhaustive/${filename}.out" --wrap="python3 $CUSTOM_FUNCTION_PATH $file"

done
