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
    This script is used to insert media files into the queue to be processed. It checks if the muxed media file
    exists and if it is already in the database. If it is not in the database, it inserts the media file into the queue.
    """

    count = 0
    skipped = 0
    for root, dirs, files in os.walk(base_path):
        if not dirs:

            count += 1
            mp4_fname = root.split("/")[-1] + ".mp4"
            full_path = join(root, mp4_fname)

            #  skip if the file doesn't exist
            if not os.path.exists(full_path):
                # print(f"Skipping {full_path}")
                skipped += 1
                continue

            if skipped > 0:
                print(f"Skipped {skipped} files, total processed {count}")
                skipped = 0

            # check if the media id is already in the database
            # print("root", root.split("/")[-1])
            media_id = "rts-" + root.split("/")[-1]
            query = text(
                "SELECT COUNT(1) FROM media WHERE media_id = :media_id")
            result = DataAccessObject().fetch_one(
                query, {"media_id": media_id})
            if result['count'] > 0:
                print(f"Media {media_id} already in the database")
                continue

            media_id_path = root.strip().replace("/mnt/rts/archives/", "")
            query = text(
                "INSERT INTO entries_to_queue (job_type, media_id) VALUES ('transcript', :media_id)")
            DataAccessObject().execute_query(
                query, {"media_id": media_id_path})


if __name__ == '__main__':
    main()
