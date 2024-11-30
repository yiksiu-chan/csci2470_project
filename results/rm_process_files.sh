#!/bin/bash

# directory containing the files (default is the current directory)
DIRECTORY=${1:-.}

# pattern match the process files
PATTERN="*_eval-process=*.json"

# remove matching files
echo "Removing process files in directory: $DIRECTORY"
find "$DIRECTORY" -type f -name "$PATTERN" -exec rm -v {} \;

echo "Process files removed successfully."
