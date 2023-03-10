import click
import uvicorn

from pathlib import Path
# Local imports
from rts.api.settings import Settings, get_settings, get_app, production_mode, get_public_folder_path  
# from audioverse.utils import read_jsonlines_file
from rts.api import router
from rts.utils import get_logger
from rts.metadata import build_clips_df, build_clip_index

LOG = get_logger()

app = get_app()

@app.on_event("startup")
async def startup_event():
    settings = get_settings()
    metadata_folder = settings.get_metadata_folder()
    archive_folder = settings.get_archive_folder()
    df = build_clips_df(archive_folder, metadata_folder, force=False)
    app.state.clips = build_clip_index(df)
    router.mount_routers(app, settings)


@app.on_event("shutdown")
async def on_shutdown():
    pass


def start_dev():
    settings = get_settings()
    uvicorn.run("rts.api.server:app", host='0.0.0.0', port=int(settings.app_port), reload=True)


def start_prod():
    production_mode()
    settings = get_settings()
    uvicorn.run(app, host='0.0.0.0', port=int(settings.app_port))

if __name__ == "__main__":
    start_prod()