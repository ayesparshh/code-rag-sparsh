#!/bin/bash

# Check if folder path is provided as argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide the folder path as an argument"
    echo "Usage: ./index_codebase.sh <folder_path>"
    exit 1
fi

# Get the folder path from command line argument
folder_path="$1"

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
    echo "Error: Directory '$folder_path' does not exist"
    exit 1
fi

echo "Processing the directory at $folder_path..."

# Run scripts with the folder_path
python parsing/preprocessing.py "$folder_path"

echo "Processing complete."