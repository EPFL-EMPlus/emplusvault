
import os
import logging
from pathlib import Path
from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Any, Dict, Tuple, Union

# Local imports
from rts.api.settings import Settings, get_settings, get_public_folder_path
from rts.utils import obj_from_json

# TODO: move the mount_routers function to a separate file


router = APIRouter()


class StaticFilesSym(StaticFiles):
    "subclass StaticFiles middleware to allow symlinks"

    def lookup_path(self, path) -> Tuple[str, Any]:
        for directory in self.all_directories:
            full_path = os.path.realpath(os.path.join(directory, path))
            try:
                stat_result = os.stat(full_path)
                return (full_path, stat_result)
            except FileNotFoundError:
                pass
        return ("", None)


def get_index(request: Request, settings: Settings) -> HTMLResponse:
    templates = Jinja2Templates(directory=get_public_folder_path())
    base_url = ''
    if settings.host == 'localhost':
        base_url = 'http://' + settings.host + ':' + settings.app_port
        if settings.app_prefix:
            base_url += settings.app_prefix
    else:
        base_url = 'https://' + settings.host + settings.app_prefix
    return templates.TemplateResponse("index.html", {"request": request, "base_url": base_url})


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return get_index(request, settings)


def mount_static_folder(app, route_name: str, folder_path: Union[str, Path]) -> None:
    app.mount(f"/{route_name}",
              StaticFilesSym(directory=Path(folder_path).expanduser()), name=route_name)
