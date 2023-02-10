import click
import pandas as pd

import rts.metadata
import rts.utils
import rts.io.media
import rts.features.audio
import rts.features.text
import rts.pipeline


LOCAL_RTS_DATA = "/media/data/rts/"
METADATA = LOCAL_RTS_DATA + 'metadata'
LOCAL_VIDEOS = LOCAL_RTS_DATA + 'archive'


def get_sample_df() -> pd.DataFrame:
    click.echo('Loading metadata')
    df = rts.metadata.load_metadata_hdf5(METADATA, 'rts_metadata')
    return rts.metadata.get_one_percent_sample(df)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--min_seconds', type=float, default=5, help='Minimum duration of a scene')
@click.option('--num_images', type=int, default=3, help='Number of images per scene')
@click.option('--compute_transcript', is_flag=True, help='Compute transcript')
@click.option('--force_media', is_flag=True, help='Force processing')
@click.option('--force_scene', is_flag=True, help='Force processing')
@click.option('--force_transcript', is_flag=True, help='Force processing')
def pipeline(min_seconds: int, num_images: int, compute_transcript: bool, force_media: bool, 
    force_scene: bool, force_transcript: bool) -> None:
    sample_df = get_sample_df()

    click.echo('Processing archive')
    rts.pipeline.simple_process_archive(sample_df[:1000], LOCAL_VIDEOS, min_seconds,
        num_images, compute_transcript, force_media, force_scene, force_transcript)


@cli.command()
def location() -> None:
    ts = rts.pipeline.load_all_transcripts(LOCAL_VIDEOS)
    fts = rts.features.text.build_location_df(ts)

    sample_df = get_sample_df()
    fts = rts.metadata.merge_location_df_with_metadata(sample_df, fts)

if __name__ == '__main__':
    cli()