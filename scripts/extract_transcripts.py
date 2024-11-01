import pandas as pd
from sqlalchemy import text
from tqdm import tqdm
from emv.db.dao import DataAccessObject
import emv.utils


def export_transcripts(export_dir: str, no_of_rows: int, df: pd.DataFrame, dao):
    visual = tqdm(total=no_of_rows)
    last_created_at = None
    last_media_id = None
    processed_rows = 0
    batch_size = 1000
    already_processed = set()

    while processed_rows < no_of_rows:
        # First batch
        if last_created_at is None:
            query = text("""
                SELECT 
                    feature.media_id as video_id,
                    feature.created_at,  
                    media.start_ts, 
                    media.end_ts, 
                    feature.data->'transcript' as text 
                FROM feature 
                LEFT JOIN media ON feature.media_id = media.media_id 
                WHERE feature.feature_type = 'transcript+ner' 
                AND length(feature.media_id) != 12
                ORDER BY feature.created_at, feature.media_id
                LIMIT :limit
            """)
            params = {"limit": batch_size}

        # Subsequent batches using keyset pagination
        else:
            query = text("""
                SELECT 
                    feature.media_id as video_id,
                    feature.created_at,  
                    media.start_ts, 
                    media.end_ts, 
                    feature.data->'transcript' as text 
                FROM feature 
                LEFT JOIN media ON feature.media_id = media.media_id 
                WHERE feature.feature_type = 'transcript+ner' 
                AND length(feature.media_id) != 12
                AND (feature.created_at, feature.media_id) > (:last_created_at, :last_media_id)
                ORDER BY feature.created_at, feature.media_id
                LIMIT :limit
            """)
            params = {
                "last_created_at": last_created_at,
                "last_media_id": last_media_id,
                "limit": batch_size
            }

        # Fetch and process data
        try:
            r = dao.fetch_all(query, params)
            if not r:  # No more records
                break

            # Update pagination markers
            last_created_at = r[-1]["created_at"]
            last_media_id = r[-1]["video_id"]

            # Create DataFrame and add columns
            df_media = pd.DataFrame(r)
            df_media = df_media[df_media['video_id'].str.len() > 12]

            df_media = df_media[~df_media['video_id'].isin(already_processed)]

            for media_id in df_media['video_id']:
                already_processed.add(media_id)
            if len(df_media) % 1000 != 0:
                # continue
                print(len(df_media), len(already_processed))
                print(list(already_processed)[:5])

            df_media['source'] = df_media['video_id'].str.split('-').str[1]

            # Apply transformations using vectorized operations instead of apply
            sources = df_media['source']
            df_media['title'] = df.loc[sources, 'title'].values
            df_media['collection'] = df.loc[sources, 'collection'].values
            df_media['content_type'] = df.loc[sources, 'contentType'].values
            df_media['category_name'] = df.loc[sources, 'categoryName'].values
            df_media['publish_date'] = df.loc[sources, 'publishedDate'].values

            # Export batch
            df_media.to_parquet(
                f"{export_dir}/transcript_export_{
                    processed_rows}.parquet",
                index=False
            )

            # Update progress
            processed_rows += len(r)
            visual.update(len(r))

        except Exception as e:
            raise

    visual.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, required=True)
    args = parser.parse_args()
    export_dir = args.export_dir

    df = emv.utils.dataframe_from_hdf5('/mnt/rts/archives/', 'rts_metadata')
    query = text(
        """SELECT COUNT(*) FROM feature WHERE feature_type = 'transcript+ner' AND length(media_id) != 12 LIMIT 10 ;""")
    no_of_rows = DataAccessObject().fetch_all(query)[0]['count']

    # Export transcripts
    export_transcripts(export_dir, no_of_rows, df, DataAccessObject())
