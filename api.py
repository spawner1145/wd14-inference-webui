from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Dict, List
import base64
from PIL import Image
from io import BytesIO
from threading import Lock

from main import predictor, utils

class TaggerInterrogateRequest(BaseModel):
    image: str
    model: str
    threshold: float = 0.35
    character_threshold: float = 0.85
    general_mcut_enabled: bool = False
    character_mcut_enabled: bool = False

class TaggerInterrogateResponse(BaseModel):
    caption: Dict[str, float]
    model_used: str

class InterrogatorsResponse(BaseModel):
    models: List[str]
    model_info: Dict[str, Dict[str, str]]

class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock, prefix: str = None) -> None:
        self.app = app
        self.queue_lock = queue_lock
        self.prefix = prefix

        self.add_api_route(
            "interrogate",
            self.endpoint_interrogate,
            methods=["POST"],
            response_model=TaggerInterrogateResponse
        )

        self.add_api_route(
            "interrogators",
            self.endpoint_interrogators,
            methods=["GET"],
            response_model=InterrogatorsResponse
        )

    def add_api_route(self, path: str, endpoint: callable, **kwargs):
        full_path = f"{self.prefix}/{path}" if self.prefix else path
        return self.app.add_api_route(full_path, endpoint,** kwargs)

    def endpoint_interrogate(self, req: TaggerInterrogateRequest):
        if not req.image:
            raise HTTPException(status_code=400, detail="Image is required")
        
        try:
            model = predictor.resolve_model(req.model)
        except ValueError as e:
            raise HTTPException(
                status_code=404,
                detail=str(e)
            )

        try:
            if req.image.startswith(('data:', ';base64,')):
                req.image = req.image.split(';base64,')[-1]
                
            image_data = base64.b64decode(req.image)
            image = Image.open(BytesIO(image_data)).convert("RGBA")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

        with self.queue_lock:
            general_tags_str, ratings, character_tags, general_tags = predictor.predict(
                image=image,
                model_name=req.model,
                general_thresh=req.threshold,
                general_mcut_enabled=req.general_mcut_enabled,
                character_thresh=req.character_threshold,
                character_mcut_enabled=req.character_mcut_enabled
            )

        caption = {**ratings,** general_tags, **character_tags}
        return TaggerInterrogateResponse(
            caption=caption,
            model_used=model.repo_id
        )

    def endpoint_interrogators(self):
        model_info = {}
        for name in utils.interrogators:
            model = utils.interrogators[name]
            model_info[name] = {
                "repo_id": model.repo_id,
                "revision": model.revision or "latest"
            }
        
        return InterrogatorsResponse(
            models=list(utils.interrogators.keys()),
            model_info=model_info
        )
