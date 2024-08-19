from emv.pipelines.utils import create_optimized_media
import os
import click
from emv.db.dao import DataAccessObject
from sqlalchemy.sql import text
from os.path import join


@click.command()
@click.option('--base_path', default="/mnt/rts/archives/2/", help='Base path to the media files')
def main(base_path):
    """
    This script is used to create optimized media files for all media files in the base_path.
    This is how the script is executed:
    python scripts/mux_archive.py --base_path /mnt/rts/archives/2/
    """

    count = 0
    skipped = 0
    # find all underlying folders and execute create_optimized_media on them
    for root, dirs, files in os.walk(base_path):
        if not dirs:
            # print(f"Processed {count} files. Current file {root}")
            # /mnt/rts/archives/6/8/1/ZT010186
            count += 1
            # print(f"Processing {root}")

            mp4_fname = root.split("/")[-1] + ".mp4"
            full_path = join(root, mp4_fname)
            if os.path.exists(full_path):
                # print(f"Skipping {full_path}")
                skipped += 1
                continue

            if skipped > 0:
                print(f"Skipped {skipped} files, total processed {count}")
                skipped = 0

            try:
                r = create_optimized_media(root, root, False)
            except Exception as e:
                print(f"Error processing {root}: {e}")
                continue

            media_id_path = root.strip().replace("/mnt/rts/archives/", "")
            query = text(
                "INSERT INTO entries_to_queue (job_type, media_id) VALUES ('transcript', :media_id)")
            DataAccessObject().execute_query(
                query, {"media_id": media_id_path})


if __name__ == '__main__':
    main()
