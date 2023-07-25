import click
import pandas as pd

import rts.utils
import rts.io.media
import rts.features.audio
import rts.features.text
import rts.pipelines.rts

from rts.db.dao import DataAccessObject
from rts.settings import DATABASE_URL, DB_HOST, DB_NAME, DB_PORT, SUPERUSER_CLI_KEY
from rts.db.utils import create_database
from rts.db.queries import create_library, create_new_user
from rts.api.models import LibraryCreate
from rts.api.routers.auth_router import UserInDB as User

from rts.pipelines.rts import RTS_ROOT_FOLDER, RTS_LOCAL_VIDEOS


@click.group()
def cli():
    pass


@cli.group()
def rts():
    pass


@cli.group()
def ioc():
    pass


@cli.group()
def mjf():
    pass


@cli.group()
def db():
    pass


@rts.command()
@click.option('--continuous', is_flag=True, help='Merge continuous sentences from the same speaker')
@click.option('--compute-transcript', is_flag=True, help='Compute transcript')
@click.option('--compute-clips', is_flag=True, help='Create clips from transcript')
@click.option('--force-media', is_flag=True, help='Force media remuxing')
@click.option('--force-transcript', is_flag=True, help='Force transcription')
@click.option('--force-clips', is_flag=True, help='Force clip extraction')
def pipeline(continuous: bool,
             compute_transcript: bool, compute_clips: bool, force_media: bool,
             force_transcript: bool, force_clips: bool) -> None:

    df = rts.pipelines.rts.get_sample_df()
    # df = get_aivectors_df()

    click.echo('Processing archive')
    click.echo(
        f'Compute transcript: {compute_transcript}, Compute clips: {compute_clips}, ')
    rts.pipelines.rts.simple_process_archive(df, RTS_LOCAL_VIDEOS, continuous,
                                             compute_transcript, compute_clips,
                                             force_media, force_transcript, force_clips)


# @rts.command()
# @click.option('--tile-size', type=int, default=64, help='Tile size')
# @click.option('--name', type=str, default='0', help='Atlas name')
# def atlas(tile_size: int, name: str) -> None:
#     df = rts.pipelines.rts.build_clips_df(RTS_ROOT_FOLDER, force=True)
#     rts.pipelines.rts.create_clip_texture_atlases(df, RTS_ROOT_FOLDER,
#                                              name,  # folder name
#                                              tile_size=tile_size,
#                                              flip=True,
#                                              no_border=True,
#                                              format='jpg')


# @rts.command()
# def location() -> None:
#     ts = rts.pipelines.rts.load_all_transcripts(RTS_LOCAL_VIDEOS)
#     fts = rts.features.text.build_location_df(ts)

#     sample_df = rts.pipelines.rts.get_sample_df()
#     fts = rts.pipelines.rts.merge_location_df_with_metadata(sample_df, fts)


@db.command()
def init_db():
    click.echo(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    DataAccessObject().connect(DATABASE_URL)

    confirm = click.prompt(
        'Are you sure you want to initialize the database? This will overwrite the current database [SUPERUSER_CLI_KEY]')
    if confirm.lower() == SUPERUSER_CLI_KEY:
        click.echo('Initializing database...')
        create_database("db/tables.sql")
        click.echo('Database has been successfully initialized.')
    else:
        click.echo('Database initialization has been canceled.')


@db.command()
@click.option('--library-name', type=str, default='rts', help='Library name')
@click.option('--version', type=str, default='0.0.1', help='Library version')
@click.option('--data', type=str, default='{}', help='Library data')
def new_library(library_name: str, version: str, data: str):
    click.echo(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    DataAccessObject().connect(DATABASE_URL)

    create_library(LibraryCreate(
        library_name=library_name,
        version=version,
        data=data
    ))
    click.echo('New library has been successfully created.')


@db.command()
def new_sample_project():
    click.echo(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    DataAccessObject().connect(DATABASE_URL)

    confirm = click.prompt(
        'Are you sure you want to create a new sample project on the database? This will add data to the database database [yes/no]')
    if confirm.lower() == 'yes':
        click.echo('Creating a new sample project...')

        # Create library
        create_library(LibraryCreate(
            library_name="sample",
            version="0.0.1",
            prefix_path="/media/data/rts",
            data='{}'
        ))

        click.echo('New sample project has been successfully created.')
    else:
        click.echo('Project creation has been canceled.')


@db.command()
@click.option('--username', type=str, default='rts', help='User name')
@click.option('--full-name', type=str, default='rts', help='User full name')
@click.option('--email', type=str, default='emplus@epfl.ch', help='User email')
def create_user(username: str, full_name: str, email: str):
    click.echo(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    DataAccessObject().connect(DATABASE_URL)

    password = click.prompt('Set password', hide_input=True,
                            confirmation_prompt=True)

    create_new_user(User(
        username=username,
        full_name=full_name,
        password=password,
        email=email
    ))


if __name__ == '__main__':
    cli()
