import argparse

import gradio as gr
import uvicorn
from fastapi import FastAPI

from backend_tagger.api import Api
from backend_tagger.runtime import queue_lock
from backend_tagger.ui import create_ui


def parse_args():
    parser = argparse.ArgumentParser(description="WaifuDiffusion Tagger service")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    return parser.parse_args()


def create_app() -> FastAPI:
    app = FastAPI(
        title="WaifuDiffusion Tagger API",
        description="Image tag extraction API",
        version="1.0.0",
    )
    Api(app, queue_lock, prefix="/tagger/v1")
    return gr.mount_gradio_app(app, create_ui(), path="")


def run() -> None:
    args = parse_args()
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
