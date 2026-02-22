#!/bin/bash

# Target directory
TARGET_DIR="data/processed_validation"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR does not exist."
    exit 1
fi

count=0

echo "Scanning for empty patches folders..."

# Loop through all subdirectories in the target directory
for dir in "$TARGET_DIR"/*/; do
    # Remove the trailing slash for cleaner output
    dir=${dir%/}
    
    # Check if the patches folder exists inside the sample directory
    if [ -d "$dir/patches" ]; then
        # Check if the patches directory is empty
        if [ -z "$(ls -A "$dir/patches")" ]; then
            echo "Removing $dir (empty patches folder)"
            rm -rf "$dir"
            ((count++))
        fi
    fi
done

echo "Cleanup complete! Removed $count empty directories."
