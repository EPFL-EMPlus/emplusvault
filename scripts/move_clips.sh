#!/bin/bash

# Set the base directory
base_dir="/media/data/rts/archive"

# Find all directories containing a "clips" folder
find "$base_dir" -type d -name "clips" | while read -r clips_dir; do
    # Check if the subdirectory "clips/clips" exists
    if [ -d "$clips_dir/clips" ]; then
        # Move all files from the subdirectory to the main clips directory
        mv "$clips_dir/clips"/* "$clips_dir/"

        # Remove the now empty subdirectory
        rmdir "$clips_dir/clips"
    fi
done

echo "All files have been moved successfully."
