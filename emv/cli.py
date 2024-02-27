import os
import click
import subprocess
import pandas as pd

import emv.utils
import emv.io.media
import emv.features.audio
import emv.features.text
import emv.pipelines.rts
import emv.preprocess.ioc

from datetime import datetime
from pathlib import Path

from emv.settings import DB_HOST, DB_NAME, DB_PORT, SUPERUSER_CLI_KEY, DB_USER, DB_PASSWORD
from emv.settings import IOC_ROOT_FOLDER


from emv.db.utils import create_database
from emv.db.queries import create_library, create_new_user, allow_user_to_access_library
from emv.api.models import LibraryCreate
from emv.api.routers.auth_router import UserInDB as User

from emv.pipelines.rts import RTS_LOCAL_VIDEOS


@click.group()
def cli():
    pass


@cli.group()
def rts_archive():
    pass


@cli.group()
def ioc_archive():
    pass


@cli.group()
def mjf_archive():
    pass


@cli.group()
def db():
    pass


@rts_archive.command()
@click.option('--continuous', is_flag=True, help='Merge continuous sentences from the same speaker')
@click.option('--compute-transcript', is_flag=True, help='Compute transcript')
@click.option('--compute-clips', is_flag=True, help='Create clips from transcript')
@click.option('--force-media', is_flag=True, help='Force media remuxing')
@click.option('--force-transcript', is_flag=True, help='Force transcription')
@click.option('--force-clips', is_flag=True, help='Force clip extraction')
def pipeline(continuous: bool,
             compute_transcript: bool, compute_clips: bool, force_media: bool,
             force_transcript: bool, force_clips: bool) -> None:

    df = emv.pipelines.rts.get_sample_df()
    # df = get_aivectors_df()

    click.echo('Processing archive')
    click.echo(
        f'Compute transcript: {compute_transcript}, Compute clips: {compute_clips}, ')
    emv.pipelines.rts.simple_process_archive(df, RTS_LOCAL_VIDEOS, continuous,
                                             compute_transcript, compute_clips,
                                             force_media, force_transcript, force_clips)


# @rts.command()
# def location() -> None:
#     ts = rts.pipelines.rts.load_all_transcripts(RTS_LOCAL_VIDEOS)
#     fts = rts.features.text.build_location_df(ts)

#     sample_df = rts.pipelines.rts.get_sample_df()
#     fts = rts.pipelines.rts.merge_location_df_with_metadata(sample_df, fts)


@db.command()
def init_db():
    click.echo(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}")
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
@click.option('--prefix_path', type=str, default='/media/data/rts/archive/', help='Library prefix path')
@click.option('--data', type=str, default='{}', help='Library data')
def new_library(library_name: str, version: str, prefix_path: str, data: str):
    click.echo(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    if prefix_path[-1] != '/':
        prefix_path += '/'

    create_library(LibraryCreate(
        library_name=library_name,
        version=version,
        prefix_path=prefix_path,
        data=data
    ))
    click.echo('New library has been successfully created.')


@db.command()
def new_sample_project():
    click.echo(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    confirm = click.prompt(
        'Are you sure you want to create a new sample project on the database? This will add data to the database database [yes/no]')
    if confirm.lower() == 'yes':
        click.echo('Creating a new sample project...')

        # Create library
        create_library(LibraryCreate(
            library_name="sample",
            version="0.0.1",
            prefix_path="/media/data/rts/archive/",
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
    password = click.prompt('Set password', hide_input=True,
                            confirmation_prompt=True)

    create_new_user(User(
        username=username,
        full_name=full_name,
        password=password,
        email=email
    ))
    click.echo('New user has been successfully created.')


@db.command()
@click.option('--user-id', type=int, default=1, help='User id')
@click.option('--library-id', type=int, default=1, help='Library id')
def allow_library_access(user_id: int, library_id: int):
    allow_user_to_access_library(user_id, library_id)
    click.echo('User has been granted access to the library.')


@db.command()
@click.option('--data', type=str, default='{}', help='csv dataframe')
def ingest_ioc(data: str):
    from emv.pipelines.ioc import PipelineIOC
    df = pd.read_csv(data)

    def find_mp4_files(root_folder):
        import os
        mp4_files = []

        for foldername, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith('.mp4'):
                    mp4_files.append(os.path.join(foldername, filename))

        return mp4_files

    mp4_files = find_mp4_files(os.path.join(IOC_ROOT_FOLDER, "videos/"))
    mp4_files += find_mp4_files("/mnt/ioc/")
    file_map = {}
    for mp4 in mp4_files:
        file_map[mp4.split('/')[-1].split('.')[0]] = mp4

    def get_path(guid):
        return file_map[guid] if guid in file_map else None
    df['path'] = df.guid.apply(get_path)

    click.echo(f"Processing {len(df)} rows")
    pipeline = PipelineIOC()
    pipeline.ingest(df)


@db.command()
@click.option('--path', type=str, default='/media/data/dumps/', help='Path to exported file or path to export folder')
def export_db(path: str):
    click.echo(f"Exporting {DB_HOST}:{DB_PORT}/{DB_NAME}")
    # Check if path is a folder or not
    if path.endswith('/') or not os.path.splitext(path)[1]:
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Set output file name
        # Create parent folder if needed
        os.makedirs(path, exist_ok=True)
        output_file = f"output_{timestamp}.sql"
        # Combine directory and file name
        path = os.path.join(path, output_file)
    else:
        # Use given path assuming it ends with .sql
        if not path.endswith('.sql'):
            path = os.path.splitext(path)[0] + '.sql'
        # Create parent folder if needed
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    user = DB_USER
    database = DB_NAME
    os.environ['PGPASSWORD'] = DB_PASSWORD

    command_get_container = 'docker ps --format "{{.CreatedAt}} {{.Names}}" | grep "docker-postgres" | sort -r | head -n 1 | awk \'{print $NF}\''
    container_name = subprocess.check_output(
        command_get_container, shell=True).decode().strip()

    # # Prepare command
    command = f'docker exec -i {container_name} pg_dump -U postgres -F p -N tiger {database} > {path}'

    # Execute command
    subprocess.run(command, shell=True, check=True)
    click.echo(f"Database dump completed. Output written to {path}")


@ioc_archive.command()
def build_video_metadata():
    click.echo('Building video metadata...')
    metadata = Path(IOC_ROOT_FOLDER) / 'metadata'
    outdir = Path(IOC_ROOT_FOLDER) / 'data'
    video_folder = Path(IOC_ROOT_FOLDER) / 'videos'
    df = emv.preprocess.ioc.create_df_from_xml(metadata, outdir, force=False)
    df = emv.preprocess.ioc.get_sport_df(df)
    df = emv.preprocess.ioc.match_video_files(df, video_folder)
    emv.utils.dataframe_to_hdf5(outdir, 'ioc_consolidated', df)


if __name__ == '__main__':
    cli()
