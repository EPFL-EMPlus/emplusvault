from pathlib import Path
from functools import lru_cache
from pydantic import BaseSettings
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost",
    "https://localhost",
    "http://localhost:8080",
    "http://localhost:9999",
    "https://localhost:9999",
    "https://dev.cables.gl",
    "https://cables.gl",
    "https://sandbox.cables.gl",
    "https://emplusdemo.epfl.ch",
    "http://emplusdemo.epfl.ch"
    # "*" # all origins
]


def get_project_root_path() -> Path:
    current_file_dir = Path(__file__).parent.parent
    project_root = current_file_dir.parent
    project_root_absolute = project_root.resolve()
    return project_root_absolute


class Settings(BaseSettings):
    library_id: int = -1
    data_folder: str = "/media/data/rts/"
    library_name: str = 'default'
    app_prefix: str = ''
    mode: str = ''
    host: str = ''
    app_port: str = '9999'

    class Config:
        env_prefix = "rts_"
        env_file = get_project_root_path() / ".env"
        env_file_encoding = 'utf-8'

    def get_metadata_folder(self) -> Path:
        return Path(self.data_folder) / "metadata"

    def get_archive_folder(self) -> Path:
        return Path(self.data_folder) / "archive"

    def get_atlases_folder(self) -> Path:
        return Path(self.data_folder) / "atlases"


class DevSettings(Settings):
    mode = 'dev'
    host = 'localhost'


class ProductionSettings(Settings):
    mode = 'production'
    host = 'emplusdemo.epfl.ch'


settings = DevSettings()


def production_mode() -> Settings:
    global settings
    settings = ProductionSettings()


def init_library(library_id: int, is_prod: bool = False) -> None:
    global settings
    if is_prod:
        production_mode()
    settings.library_id = library_id


@lru_cache()
def get_settings() -> Settings:
    return settings


def get_public_folder_path() -> Path:
    path = get_project_root_path() / "public"
    return path


app = FastAPI(default_response_class=ORJSONResponse)
app.add_middleware(GZipMiddleware, minimum_size=3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex='https://.*\.cables\.gl',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # expose_headers=["*"]
)


def get_app():
    return app
