import argparse
import os
import shutil
from pathlib import Path
from threading import Lock
import gradio as gr
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
from fastapi import FastAPI
import uvicorn
import json

class WaifuDiffusionInterrogator:
    def __init__(self, display_name: str, repo_id: str, revision: str = None, subfolder: str = None,
                 model_type: str = 'default', use_optimized_model: bool = False):
        self.display_name = display_name
        self.repo_id = repo_id
        self.revision = revision
        self.subfolder = subfolder
        self.model_type = model_type
        self.use_optimized_model = use_optimized_model
        self.model_target_size = None
        self.model = None
        self.tag_names = []
        self.rating_indexes, self.general_indexes, self.character_indexes = [], [], []
        self.artist_indexes, self.copyright_indexes, self.meta_indexes, self.quality_indexes = [], [], [], []

    def get_model_dir(self, base_dir: Path) -> Path:
        repo_name = self.repo_id.replace("/", "_")
        if self.revision: repo_name += f"_{self.revision}"
        if self.subfolder: repo_name += f"_{self.subfolder.replace('/', '_')}"
        if self.use_optimized_model: repo_name += "_optimized"
        return base_dir / repo_name

class Utils:
    def __init__(self):
        self.interrogators = {
            'wd14-convnextv2-v2': WaifuDiffusionInterrogator('wd14-convnextv2-v2', repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2', revision='v2.0'),
            'wd14-vit-v2': WaifuDiffusionInterrogator('wd14-vit-v2', repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2', revision='v2.0'),
            'wd14-convnext-v2': WaifuDiffusionInterrogator('wd14-convnext-v2', repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2', revision='v2.0'),
            'wd14-swinv2-v2': WaifuDiffusionInterrogator('wd14-swinv2-v2', repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2', revision='v2.0'),
            'wd14-convnextv2-v2-git': WaifuDiffusionInterrogator('wd14-convnextv2-v2-git', repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2'),
            'wd14-vit-v2-git': WaifuDiffusionInterrogator('wd14-vit-v2-git', repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2'),
            'wd14-convnext-v2-git': WaifuDiffusionInterrogator('wd14-convnext-v2-git', repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2'),
            'wd14-swinv2-v2-git': WaifuDiffusionInterrogator('wd14-swinv2-v2-git', repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2'),
            'wd14-vit': WaifuDiffusionInterrogator('wd14-vit', repo_id='SmilingWolf/wd-v1-4-vit-tagger'),
            'wd14-convnext': WaifuDiffusionInterrogator('wd14-convnext', repo_id='SmilingWolf/wd-v1-4-convnext-tagger'),
            'wd14-vit-v3-git': WaifuDiffusionInterrogator('wd14-vit-v3-git', repo_id='SmilingWolf/wd-vit-tagger-v3'),
            'wd14-convnext-v3-git': WaifuDiffusionInterrogator('wd14-convnext-v3-git', repo_id='SmilingWolf/wd-convnext-tagger-v3'),
            'wd14-swinv2-v3-git': WaifuDiffusionInterrogator('wd14-swinv2-v3-git', repo_id='SmilingWolf/wd-swinv2-tagger-v3'),
            'wd14-large-v3-git': WaifuDiffusionInterrogator('wd14-large-v3-git', repo_id='SmilingWolf/wd-vit-large-tagger-v3'),
            'wd14-eva02-large-v3-git': WaifuDiffusionInterrogator('wd14-eva02-large-v3-git', repo_id='SmilingWolf/wd-eva02-large-tagger-v3'),
            'idolsankaku-swinv2-tagger-v1': WaifuDiffusionInterrogator('idolsankaku-swinv2-tagger-v1', repo_id='deepghs/idolsankaku-swinv2-tagger-v1'),
            'idolsankaku-eva02-large-tagger-v1': WaifuDiffusionInterrogator('idolsankaku-eva02-large-tagger-v1', repo_id='deepghs/idolsankaku-eva02-large-tagger-v1'),
            'wd-v1-4-moat-tagger-v2': WaifuDiffusionInterrogator('wd-v1-4-moat-tagger-v2', repo_id='SmilingWolf/wd-v1-4-moat-tagger-v2'),
            'cltagger-v1.02': WaifuDiffusionInterrogator('cltagger-v1.02', repo_id='cella110n/cl_tagger', subfolder='cl_tagger_1_02', model_type='cl_tagger', use_optimized_model=False),
            'cltagger-v1.02-optimized': WaifuDiffusionInterrogator('cltagger-v1.02-optimized', repo_id='cella110n/cl_tagger', subfolder='cl_tagger_1_02', model_type='cl_tagger', use_optimized_model=True),
        }
        self.kaomojis = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"]
        self.all_model_names = list(self.interrogators.keys())

utils = Utils()
queue_lock = Lock()
MODEL_BASE_DIR = Path(__file__).parent / "models"
MODEL_BASE_DIR.mkdir(exist_ok=True)

class Predictor:
    def __init__(self):
        self.last_loaded_model = None

    def resolve_model(self, model_name: str) -> WaifuDiffusionInterrogator:
        if model_name not in utils.interrogators:
            raise ValueError(f"模型不存在: {model_name}\n可用模型: {list(utils.interrogators.keys())}")
        return utils.interrogators[model_name]

    def download_model(self, model: WaifuDiffusionInterrogator):
        model_dir = model.get_model_dir(MODEL_BASE_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename, label_csv, label_json = "model.onnx", "selected_tags.csv", "tag_mapping.json"
        
        if model.model_type == 'cl_tagger':
            model_filename = "model_optimized.onnx" if model.use_optimized_model else "model.onnx"
            allowed_patterns = [f"{model.subfolder}/{f}" for f in [model_filename, label_json]]
        else:
            allowed_patterns = [f"{model.subfolder}/{p}" if model.subfolder else p for p in [model_filename, label_csv]]

        model_path, label_path_csv, label_path_json = model_dir / model_filename, model_dir / label_csv, model_dir / label_json
        
        if not model_path.exists() or not (label_path_csv.exists() or label_path_json.exists()):
            print(f"下载模型文件: {model_filename} from {model.repo_id}/{model.subfolder or ''}")
            temp_dir = model_dir / "temp_download"
            try:
                huggingface_hub.snapshot_download(
                    repo_id=model.repo_id, revision=model.revision, local_dir=temp_dir, allow_patterns=allowed_patterns,
                    use_auth_token=os.environ.get("HF_TOKEN"), local_dir_use_symlinks=False
                )
                src_base = temp_dir / model.subfolder if model.subfolder else temp_dir
                if (src_base / model_filename).exists(): shutil.move(str(src_base / model_filename), str(model_path))
                if (src_base / label_csv).exists(): shutil.move(str(src_base / label_csv), str(label_path_csv))
                if (src_base / label_json).exists(): shutil.move(str(src_base / label_json), str(label_path_json))
            finally:
                if temp_dir.exists(): shutil.rmtree(temp_dir, ignore_errors=True)

        final_label_path = None
        if label_path_csv.exists(): final_label_path = label_path_csv
        elif label_path_json.exists(): final_label_path = label_path_json
        else: raise FileNotFoundError(f"标签文件未在 {model_dir} 中找到")
            
        return final_label_path, model_path

    def load_model(self, model: WaifuDiffusionInterrogator):
        if model == self.last_loaded_model: return
        with queue_lock:
            if model == self.last_loaded_model: return
            label_path, model_path = self.download_model(model)
            if model.model_type == 'cl_tagger': self.load_labels_cl(model, label_path)
            else: self.load_labels_default(model, label_path)

            if model.model: del model.model
            model.model = rt.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            
            if model.model_type == 'cl_tagger':
                model.model_target_size = 448
            else:
                input_shape = model.model.get_inputs()[0].shape
                model.model_target_size = input_shape[1]

            self.last_loaded_model = model
            print(f"已加载模型: {model.display_name} (类型: {model.model_type})")

    def load_labels_default(self, model, csv_path):
        dataframe = pd.read_csv(csv_path)
        model.tag_names = dataframe["name"].map(lambda x: x.replace("_", " ") if x not in utils.kaomojis else x).tolist()
        model.rating_indexes = list(np.where(dataframe["category"] == 9)[0])
        model.general_indexes = list(np.where(dataframe["category"] == 0)[0])
        model.character_indexes = list(np.where(dataframe["category"] == 4)[0])

    def load_labels_cl(self, model, json_path):
        with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        if "idx_to_tag" in data:
            idx_map, cat_map = ({int(k): v for k, v in data["idx_to_tag"].items()}, data["tag_to_category"])
        else:
            d_int = {int(k): v for k, v in data.items()}
            idx_map, cat_map = ({i:d['tag'] for i,d in d_int.items()}, {d['tag']:d['category'] for d in d_int.values()})
        
        max_idx = max(idx_map.keys()); model.tag_names = [""] * (max_idx + 1)
        for idx, tag in idx_map.items():
            model.tag_names[idx] = tag.replace("_", " ")
            category = cat_map.get(tag, 'Unknown')
            if category == 'Rating': model.rating_indexes.append(idx)
            elif category == 'General': model.general_indexes.append(idx)
            elif category == 'Artist': model.artist_indexes.append(idx)
            elif category == 'Character': model.character_indexes.append(idx)
            elif category == 'Copyright': model.copyright_indexes.append(idx)
            elif category == 'Meta': model.meta_indexes.append(idx)
            elif category == 'Quality': model.quality_indexes.append(idx)

    def prepare_image_default(self, image, target_size):
        canvas = Image.new("RGBA", image.size, (255, 255, 255)); canvas.alpha_composite(image); image = canvas.convert("RGB")
        max_dim = max(image.size)
        pad_l, pad_t = (max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255)); padded.paste(image, (pad_l, pad_t))
        if max_dim != target_size: padded = padded.resize((target_size, target_size), Image.BICUBIC)
        img_array = np.asarray(padded, dtype=np.float32)[:, :, ::-1]
        return np.expand_dims(img_array, axis=0)

    def prepare_image_cl(self, image, target_size):
        if image.mode not in ["RGB", "RGBA"]: image = image.convert("RGB")
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255)); bg.paste(image, mask=image.split()[3]); image = bg
        w, h = image.size
        if w != h:
            s = max(w, h); new_img = Image.new("RGB", (s, s), (255, 255, 255))
            new_img.paste(image, ((s - w) // 2, (s - h) // 2)); image = new_img
        img_arr = np.array(image.resize((target_size, target_size), Image.BICUBIC), dtype=np.float32) / 255.0
        img_arr = img_arr.transpose(2, 0, 1)
        mean, std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1), np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        img_arr = (img_arr - mean) / std
        return np.expand_dims(img_arr, axis=0)

    def mcut_threshold(self, probs):
        s_probs = probs[probs.argsort()[::-1]]; difs = s_probs[:-1] - s_probs[1:]; t = difs.argmax()
        return (s_probs[t] + s_probs[t + 1]) / 2

    def get_tags_cl(self, probs, model, gen_thresh, char_thresh):
        preds = {}
        if model.rating_indexes:
            preds['rating'] = [(model.tag_names[i], probs[i]) for i in model.rating_indexes]
        
        if model.quality_indexes:
            p = probs[model.quality_indexes]
            idx = np.argmax(p)
            preds['quality'] = [(model.tag_names[model.quality_indexes[idx]], p[idx])]
            
        cat_map = {
            "general": (model.general_indexes, gen_thresh), "character": (model.character_indexes, char_thresh),
            "copyright": (model.copyright_indexes, char_thresh), "artist": (model.artist_indexes, char_thresh),
            "meta": (model.meta_indexes, gen_thresh)
        }
        for cat, (indices, thresh) in cat_map.items():
            if indices:
                preds[cat] = sorted([(model.tag_names[i], probs[i]) for i in indices if probs[i] > thresh], key=lambda x: x[1], reverse=True)
        return preds

    def predict(self, image, model_name, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled):
        model = self.resolve_model(model_name)
        self.load_model(model)
        
        image_tensor = self.prepare_image_cl(image, model.model_target_size) if model.model_type == 'cl_tagger' else self.prepare_image_default(image, model.model_target_size)

        input_name = model.model.get_inputs()[0].name
        label_name = model.model.get_outputs()[0].name
        preds = model.model.run([label_name], {input_name: image_tensor.astype(np.float32)})[0][0]
        
        if model.model_type == 'cl_tagger':
            probs = 1 / (1 + np.exp(-np.clip(preds, -30, 30)))
            predictions = self.get_tags_cl(probs, model, general_thresh, character_thresh)

            ratings = dict(predictions.get("rating", []))
            quality = dict(predictions.get("quality", []))
            characters = dict(predictions.get("character", []))
            artist = dict(predictions.get("artist", []))
            copyright = dict(predictions.get("copyright", []))
            meta = dict(predictions.get("meta", []))
            general_tags = dict(predictions.get("general", []))

            combined_tags = {
                **general_tags, 
                **characters, 
                **copyright, 
                **artist, 
                **meta
            }

            sorted_tags = sorted(combined_tags.items(), key=lambda x: x[1], reverse=True)
            sorted_str = ",".join([tag for tag, prob in sorted_tags]).replace("(", r"\(").replace(")", r"\)")
            return sorted_str, ratings, quality, characters, artist, copyright, meta, general_tags

        else: # default logic
            labels = list(zip(model.tag_names, preds.astype(float)))
            ratings = dict([labels[i] for i in model.rating_indexes])
            
            general_tags_list = [labels[i] for i in model.general_indexes]
            if general_mcut_enabled and len(general_tags_list) > 1:
                general_thresh = self.mcut_threshold(np.array([x[1] for x in general_tags_list]))
            general_tags = dict([x for x in general_tags_list if x[1] > general_thresh])

            character_tags_list = [labels[i] for i in model.character_indexes]
            if character_mcut_enabled and len(character_tags_list) > 1:
                character_thresh = self.mcut_threshold(np.array([x[1] for x in character_tags_list]))
            character_tags = dict([x for x in character_tags_list if x[1] > character_thresh])
            
            combined_tags = {**general_tags, **character_tags}
            sorted_tags = sorted(combined_tags.items(), key=lambda x: x[1], reverse=True)
            sorted_str = ",".join([x[0] for x in sorted_tags]).replace("(", r"\(").replace(")", r"\)")
            
            return sorted_str, ratings, {}, character_tags, {}, {}, {}, combined_tags

predictor = Predictor()

def create_gradio_interface():
    TITLE = "WaifuDiffusion Tagger"
    DESCRIPTION = """
    <br>
    - API文档: <code>/docs</code> 或 <code>/redoc</code><br>
    - API接口: <code>/tagger/v1</code>
    """
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"<h1 style='text-align: center'>{TITLE}</h1>"); gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column(variant="panel"):
                image = gr.Image(type="pil", image_mode="RGBA", label="输入图像")
                model = gr.Dropdown(choices=utils.all_model_names, label="选择模型", value="wd14-eva02-large-v3-git")
                with gr.Row():
                    general_thresh = gr.Slider(0, 1, 0.35, 0.05, label="通用标签阈值")
                    general_mcut = gr.Checkbox(False, label="使用MCut自动阈值（通用）")
                with gr.Row():
                    char_thresh = gr.Slider(0, 1, 0.85, 0.05, label="角色/作者/版权 阈值")
                    char_mcut = gr.Checkbox(False, label="使用MCut自动阈值（角色）")
                with gr.Row():
                    clear = gr.ClearButton(variant="secondary")
                    submit = gr.Button("提取标签", variant="primary")
            with gr.Column(variant="panel"):
                output_str = gr.Textbox(label="格式化标签（逗号分隔）", lines=5)
                with gr.Row():
                    ratings_out = gr.Label(label="内容评分", num_top_classes=4)
                    quality_out = gr.Label(label="质量标签")
                with gr.Row():
                    artist_out = gr.Label(label="作者")
                    copyright_out = gr.Label(label="作品系列")
                characters_out = gr.Label(label="角色识别", num_top_classes=5)
                with gr.Row():
                    meta_out = gr.Label(label="元数据")
                    general_out = gr.Label(label="通用标签")
        
        inputs = [image, model, general_thresh, general_mcut, char_thresh, char_mcut]
        outputs = [output_str, ratings_out, quality_out, characters_out, artist_out, copyright_out, meta_out, general_out]
        
        clear.click(lambda: [None, utils.all_model_names[0], 0.35, False, 0.85, False] + [None] * len(outputs), outputs=inputs + outputs, show_progress=False)
        submit.click(fn=predictor.predict, inputs=inputs, outputs=outputs)
    return demo

def parse_args():
    parser = argparse.ArgumentParser(description="WaifuDiffusion Tagger 服务")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务绑定的主机地址")
    parser.add_argument("--port", type=int, default=7860, help="服务监听的端口号")
    return parser.parse_args()

def run_server():
    args = parse_args()
    app = FastAPI(title="WaifuDiffusion Tagger API", description="图像标签提取工具的API接口", version="1.0.0")

    try:
        from api import Api
        Api(app, queue_lock, prefix="/tagger/v1")
        print("API endpoints loaded from api.py")
    except ImportError:
        print("api.py not found, running in Gradio-only mode.")
    except Exception as e:
        print(f"Error loading api.py: {e}")

    gradio_app = create_gradio_interface()
    app = gr.mount_gradio_app(app, gradio_app, path="")

    print(f"服务启动信息:")
    print(f"- Gradio界面: http://{args.host}:{args.port}")
    print(f"- API文档: http://{args.host}:{args.port}/docs 或 http://{args.host}:{args.port}/redoc")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    run_server()
