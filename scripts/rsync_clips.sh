#!/bin/bash

# Set the source and destination directories
ROOT_SOURCE_DIR="/media/data/rts"
SOURCE_DIR="$ROOT_SOURCE_DIR/archive"

ROOT_DEST_DIR="/media/data/rts/sample"  #/media/sinergia/RTS/sample
DEST_DIR="$ROOT_DEST_DIR/archive"

# Find all subdirectories named "clips" and use rsync to copy them along with the specified JSON files to the destination
find "$SOURCE_DIR" -type d -name 'clips' -exec bash -c '
    dest_path="${1%/}/$(realpath --relative-to="$2" "$(dirname "$0")")"
    sudo mkdir -p "$dest_path"
    
    # Copy the "clips" directory
    sudo rsync -av --no-compress "${0}" "$dest_path"
    
    # Copy the "clips.json" and "media.json" files if they exist
    for json_file in clips.json media.json; do
        if [ -f "$(dirname "$0")/$json_file" ]; then
            sudo rsync -av --no-compress "$(dirname "$0")/$json_file" "$dest_path/$json_file"
        fi
    done
' '{}' "$DEST_DIR" "$SOURCE_DIR" \;

sudo rsync -av --no-compress "$ROOT_SOURCE_DIR/metadata/" "$ROOT_DEST_DIR/metadata/"
sudo rsync -av --no-compress "$ROOT_SOURCE_DIR/atlases/" "$ROOT_DEST_DIR/atlases/"