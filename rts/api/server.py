import click
import uvicorn

from fastapi.staticfiles import StaticFiles
from pathlib import Path
# Local imports
from rts.api.settings import Settings, get_settings, get_app, init_library, get_public_folder_path
# from audioverse.utils import read_jsonlines_file
from rts.api import router
from rts.utils import get_logger
from rts.metadata import build_clips_df, build_clip_index
from rts.db.dao import DataAccessObject
from rts.db.queries import get_library_id_from_name
from rts.api.routers.library_router import library_router
from rts.api.routers.projection_router import projection_router
from rts.api.routers.media_router import media_router
# from rts.api.router import stream_router
from rts.db_settings import DATABASE_URL, DB_HOST, DB_NAME

LOG = get_logger()
app = get_app()


@click.group()
def cli():
    pass


def mount_routers(app, settings: Settings) -> None:
    # app.include_router(stream_router)
    app.include_router(library_router, tags=["library"])
    app.include_router(projection_router, tags=["projection"])
    app.include_router(media_router, tags=["media"])
    if settings.mode == 'prod':
        mount_point = settings.app_prefix if settings.app_prefix else ''
        # print("MOUNT POINT: ", mount_point)
        app.mount(mount_point + "/",
                  StaticFiles(directory=get_public_folder_path(), html=True), name="public")
    else:
        app.mount("/", StaticFiles(directory=get_public_folder_path(),
                  html=True), name="public")


@app.on_event("startup")
async def startup_event():
    settings = get_settings()
    metadata_folder = settings.get_metadata_folder()
    archive_folder = settings.get_archive_folder()

    # connect to the database
    # dao = DataAccessObject()
    # LOG.info(f"Connecting to database: {DB_HOST}/{DB_NAME}")
    # await dao.connect(DATABASE_URL)
    try:
        # TODO: Replace the function with database calls
        df = build_clips_df(archive_folder, metadata_folder, force=False)
        app.state.clips = build_clip_index(df)
    except KeyError as e:
        pass
        # LOG.error(f'KeyError: {e}')
        # LOG.error(f'Please check that all clip files are available')
    mount_routers(app, settings)


@app.on_event("shutdown")
async def on_shutdown():
    dao = DataAccessObject()
    dao.disconnect()


@cli.command('dev', help='Start the server in development mode')
@click.option('--library', '-l', default='rts', help='Video collection (default: rts)')
def start_dev(library: str):
    DataAccessObject().connect(DATABASE_URL)
    library_id = get_library_id_from_name(library)
    if not library_id:
        LOG.error(
            f"Library {library} does not exist. Please use the cli to create a new one.")
        return

    init_library(library_id, is_prod=False)
    settings = get_settings()
    uvicorn.run("rts.api.server:app", host='0.0.0.0',
                port=int(settings.app_port), reload=True)


@cli.command('prod')
@click.option('--library', '-l', default='rts', help='Video collection (default: rts)')
def start_prod(library: str):

    DataAccessObject().connect(DATABASE_URL)
    library_id = get_library_id_from_name(library)
    if not library_id:
        LOG.error(
            f"Library {library} does not exist. Please use the cli to create a new one.")
        return

    init_library(library_id, is_prod=True)
    settings = get_settings()

    uvicorn.run(app, host='0.0.0.0', port=int(settings.app_port))


if __name__ == "__main__":
    cli()
