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


def get_aivectors_df() -> pd.DataFrame:
    click.echo('Loading AI vectors subset')
    df = rts.utils.dataframe_from_hdf5(LOCAL_RTS_DATA + '/metadata', 'rts_aivectors')
    return df


@click.group()
def cli():
    pass


@cli.command()
@click.option('--min-seconds', type=float, default=6, help='Minimum duration of a scene')
@click.option('--num-images', type=int, default=3, help='Number of images per scene')
@click.option('--compute-scenes', is_flag=True, help='Compute py detect scenes')
@click.option('--compute-transcript', is_flag=True, help='Compute transcript')
@click.option('--force-media', is_flag=True, help='Force processing')
@click.option('--force-scene', is_flag=True, help='Force processing')
@click.option('--force-transcript', is_flag=True, help='Force processing')
def pipeline(min_seconds: int, num_images: int, 
    compute_scenes: bool, compute_transcript: bool, force_media: bool, 
    force_scene: bool, force_transcript: bool) -> None:

    # df = get_sample_df()
    df = get_aivectors_df()

    click.echo('Processing archive')
    rts.pipeline.simple_process_archive(df, LOCAL_VIDEOS, min_seconds,
        num_images, compute_scenes, compute_transcript, force_media, force_scene, force_transcript)


@cli.command()
@click.option('--tile-size', type=int, default=64, help='Tile size')
@click.option('--name', type=str, default='0', help='Atlas name')
def atlas(tile_size: int, name: str) -> None:
    df = rts.metadata.build_clips_df(LOCAL_VIDEOS, METADATA, force=True)
    rts.metadata.create_clip_texture_atlases(df, LOCAL_RTS_DATA,
                                                    name, # folder name
                                                    tile_size=tile_size,
                                                    flip=True,
                                                    no_border=True, 
                                                    format='jpg')


@cli.command()
def location() -> None:
    ts = rts.pipeline.load_all_transcripts(LOCAL_VIDEOS)
    fts = rts.features.text.build_location_df(ts)

    sample_df = get_sample_df()
    fts = rts.metadata.merge_location_df_with_metadata(sample_df, fts)




if __name__ == '__main__':
    cli()