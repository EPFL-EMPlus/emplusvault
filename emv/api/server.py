import click
import uvicorn

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Depends, Form
from pathlib import Path
# Local imports
from emv.api.api_settings import Settings, get_settings, get_app, init_library, get_public_folder_path
# from audioverse.utils import read_jsonlines_file
from emv.api import router
from emv.utils import get_logger
from emv.db.dao import DataAccessObject
from emv.db.queries import get_library_id_from_name
from emv.api.routers.library_router import library_router
from emv.api.routers.projection_router import projection_router
from emv.api.routers.media_router import media_router
from emv.api.routers.feature_router import feature_router
from emv.api.routers.atlas_router import atlas_router
from emv.api.routers.stream_router import stream_router
from emv.api.routers.auth_router import auth_router
from emv.settings import DATABASE_URL


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
    app.include_router(feature_router, tags=["feature"])
    app.include_router(atlas_router, tags=["atlas"])
    app.include_router(stream_router, tags=["stream"])
    app.include_router(auth_router, tags=["auth"])

    if settings.app_mode == 'prod':
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
    dao = DataAccessObject()
    mount_routers(app, settings)


@app.on_event("shutdown")
async def on_shutdown():
    dao = DataAccessObject()


@cli.command('dev', help='Start the server in development mode')
@click.option('--library', '-l', default='rts', help='Video collection (default: rts)')
def start_dev(library: str):
    DataAccessObject().connect(DATABASE_URL)
    LOG.info("Starting server in development mode")
    # LOG.info(f"Connecting to database {DATABASE_URL}")
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
