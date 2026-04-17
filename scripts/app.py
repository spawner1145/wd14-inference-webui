from PIL import Image, ImageFile

from modules import script_callbacks

from backend_tagger.api import on_app_started
from backend_tagger.ui import on_ui_tabs


Image.init()
ImageFile.LOAD_TRUNCATED_IMAGES = True

script_callbacks.on_app_started(on_app_started)
script_callbacks.on_ui_tabs(on_ui_tabs)
