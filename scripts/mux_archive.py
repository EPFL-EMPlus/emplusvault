from emv.pipelines.utils import create_optimized_media
import os
import click

@click.command()
@click.option('--base_path', default="/mnt/rts/archives/2/", help='Base path to the media files')
def main(base_path):
    """
    This script is used to create optimized media files for all media files in the base_path.
    This is how the script is executed:
    python scripts/mux_archive.py --base_path /mnt/rts/archives/2/
    """
    
    count = 0
    # find all underlying folders and execute create_optimized_media on them
    for root, dirs, files in os.walk(base_path):
        if not dirs:
            print(f"Processed {count} files. Current file {root}")
            count += 1
            try:
                r = create_optimized_media(root, root, False)
            except Exception as e:
                print(f"Error processing {root}: {e}")
                continue

if __name__ == '__main__':
    main()