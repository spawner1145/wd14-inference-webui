import base64
from io import BytesIO
from secrets import compare_digest
from threading import Lock
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
from pydantic import BaseModel

from backend_tagger.runtime import predictor, utils

try:
    from modules import shared
except ImportError:
    shared = None


class TaggerInterrogateRequest(BaseModel):
    image: str
    model: str
    threshold: float = 0.35
    character_threshold: float = 0.85
    general_mcut_enabled: bool = False
    character_mcut_enabled: bool = False


class TaggerInterrogateResponse(BaseModel):
    caption: dict[str, float]
    model_used: str


class ModelInfo(BaseModel):
    repo_id: str
    revision: str
    subfolder: str | None = None
    model_type: str


class InterrogatorsResponse(BaseModel):
    models: list[str]
    model_info: dict[str, ModelInfo]


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock, prefix: str | None = None) -> None:
        self.app = app
        self.queue_lock = queue_lock
        self.prefix = prefix
        self.credentials = self._load_credentials()

        self.add_api_route(
            "interrogate",
            self.endpoint_interrogate,
            methods=["POST"],
            response_model=TaggerInterrogateResponse,
        )
        self.add_api_route(
            "interrogators",
            self.endpoint_interrogators,
            methods=["GET"],
            response_model=InterrogatorsResponse,
        )

    def _load_credentials(self) -> dict[str, str]:
        if shared is None or not getattr(shared.cmd_opts, "api_auth", None):
            return {}

        credentials = {}
        for auth in shared.cmd_opts.api_auth.split(","):
            user, password = auth.split(":", 1)
            credentials[user] = password
        return credentials

    def auth(self, creds: HTTPBasicCredentials = Depends(HTTPBasic())) -> bool:
        if creds.username in self.credentials and compare_digest(creds.password, self.credentials[creds.username]):
            return True
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        full_path = f"{self.prefix}/{path}" if self.prefix else f"/{path}"
        if self.credentials:
            return self.app.add_api_route(full_path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(full_path, endpoint, **kwargs)

    def endpoint_interrogate(self, req: TaggerInterrogateRequest) -> TaggerInterrogateResponse:
        if not req.image:
            raise HTTPException(status_code=400, detail="Image is required")

        try:
            model = predictor.resolve_model(req.model)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            image_string = req.image.split(";base64,", 1)[-1] if ";base64," in req.image else req.image
            image = Image.open(BytesIO(base64.b64decode(image_string))).convert("RGBA")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        with self.queue_lock:
            _caption_text, *groups = predictor.predict(
                image=image,
                model_name=req.model,
                general_thresh=req.threshold,
                general_mcut_enabled=req.general_mcut_enabled,
                character_thresh=req.character_threshold,
                character_mcut_enabled=req.character_mcut_enabled,
            )

        caption: dict[str, float] = {}
        for group in groups:
            caption.update(group)
        sorted_caption = dict(sorted(caption.items(), key=lambda item: item[1], reverse=True))
        return TaggerInterrogateResponse(caption=sorted_caption, model_used=model.repo_id)

    def endpoint_interrogators(self) -> InterrogatorsResponse:
        model_info = {
            name: ModelInfo(
                repo_id=model.repo_id,
                revision=model.revision or "latest",
                subfolder=model.subfolder,
                model_type=model.model_type,
            )
            for name, model in utils.interrogators.items()
        }
        return InterrogatorsResponse(models=list(utils.interrogators.keys()), model_info=model_info)


def on_app_started(_: object, app: FastAPI) -> None:
    from backend_tagger.runtime import queue_lock

    Api(app, queue_lock, "/tagger/v1")
