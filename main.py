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

class WaifuDiffusionInterrogator:
    def __init__(self, display_name: str, repo_id: str, revision: str = None):
        self.display_name = display_name
        self.repo_id = repo_id
        self.revision = revision
        self.model_target_size = None
        self.model = None
        self.tag_names = []
        self.rating_indexes = []
        self.general_indexes = []
        self.character_indexes = []

    def get_model_dir(self, base_dir: Path) -> Path:
        repo_name = self.repo_id.replace("/", "_")
        if self.revision:
            repo_name += f"_{self.revision}"
        return base_dir / repo_name

class Utils:
    def __init__(self):
        self.interrogators = {
            # Dataset v2 带版本
            'wd14-convnextv2-v2': WaifuDiffusionInterrogator(
                'wd14-convnextv2-v2',
                repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
                revision='v2.0'
            ),
            'wd14-vit-v2': WaifuDiffusionInterrogator(
                'wd14-vit-v2',
                repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2',
                revision='v2.0'
            ),
            'wd14-convnext-v2': WaifuDiffusionInterrogator(
                'wd14-convnext-v2',
                repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2',
                revision='v2.0'
            ),
            'wd14-swinv2-v2': WaifuDiffusionInterrogator(
                'wd14-swinv2-v2',
                repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2',
                revision='v2.0'
            ),
            
            # Dataset v2 无版本（git最新）
            'wd14-convnextv2-v2-git': WaifuDiffusionInterrogator(
                'wd14-convnextv2-v2-git',
                repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2'
            ),
            'wd14-vit-v2-git': WaifuDiffusionInterrogator(
                'wd14-vit-v2-git',
                repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2'
            ),
            'wd14-convnext-v2-git': WaifuDiffusionInterrogator(
                'wd14-convnext-v2-git',
                repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2'
            ),
            'wd14-swinv2-v2-git': WaifuDiffusionInterrogator(
                'wd14-swinv2-v2-git',
                repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2'
            ),
            
            # Dataset v1 模型
            'wd14-vit': WaifuDiffusionInterrogator(
                'wd14-vit',
                repo_id='SmilingWolf/wd-v1-4-vit-tagger'
            ),
            'wd14-convnext': WaifuDiffusionInterrogator(
                'wd14-convnext',
                repo_id='SmilingWolf/wd-v1-4-convnext-tagger'
            ),
            
            # Dataset v3 模型
            'wd14-vit-v3-git': WaifuDiffusionInterrogator(
                'wd14-vit-v3-git',
                repo_id='SmilingWolf/wd-vit-tagger-v3'
            ),
            'wd14-convnext-v3-git': WaifuDiffusionInterrogator(
                'wd14-convnext-v3-git',
                repo_id='SmilingWolf/wd-convnext-tagger-v3'
            ),
            'wd14-swinv2-v3-git': WaifuDiffusionInterrogator(
                'wd14-swinv2-v3-git',
                repo_id='SmilingWolf/wd-swinv2-tagger-v3'
            ),
            'wd14-large-v3-git': WaifuDiffusionInterrogator(
                'wd14-large-v3-git',
                repo_id='SmilingWolf/wd-vit-large-tagger-v3'
            ),
            'wd14-eva02-large-v3-git': WaifuDiffusionInterrogator(
                'wd14-eva02-large-v3-git',
                repo_id='SmilingWolf/wd-eva02-large-tagger-v3'
            ),
            
            # IdolSankaku 模型
            'idolsankaku-swinv2-tagger-v1': WaifuDiffusionInterrogator(
                'idolsankaku-swinv2-tagger-v1',
                repo_id='deepghs/idolsankaku-swinv2-tagger-v1'
            ),
            'idolsankaku-eva02-large-tagger-v1': WaifuDiffusionInterrogator(
                'idolsankaku-eva02-large-tagger-v1',
                repo_id='deepghs/idolsankaku-eva02-large-tagger-v1'
            ),
            'wd-v1-4-moat-tagger-v2': WaifuDiffusionInterrogator(
                'wd-v1-4-moat-tagger-v2',
                repo_id='SmilingWolf/wd-v1-4-moat-tagger-v2'
            )
        }

        self.kaomojis = [
            "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", 
            ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"
        ]

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
            raise ValueError(
                f"模型不存在: {model_name}\n可用模型: {list(utils.interrogators.keys())}"
            )
        return utils.interrogators[model_name]

    def download_model(self, model: WaifuDiffusionInterrogator):
        model_dir = model.get_model_dir(MODEL_BASE_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.onnx"
        label_path = model_dir / "selected_tags.csv"

        if not model_path.exists():
            print(f"下载模型: {model.repo_id} (版本: {model.revision or 'latest'})")
            temp_path = huggingface_hub.hf_hub_download(
                repo_id=model.repo_id,
                filename="model.onnx",
                revision=model.revision,
                use_auth_token=os.environ.get("HF_TOKEN")
            )
            shutil.move(temp_path, model_path)

        if not label_path.exists():
            print(f"下载标签: {model.repo_id}")
            temp_path = huggingface_hub.hf_hub_download(
                repo_id=model.repo_id,
                filename="selected_tags.csv",
                revision=model.revision,
                use_auth_token=os.environ.get("HF_TOKEN")
            )
            shutil.move(temp_path, label_path)

        return label_path, model_path

    def load_model(self, model: WaifuDiffusionInterrogator):
        if model == self.last_loaded_model:
            return

        csv_path, model_path = self.download_model(model)
        tags_df = pd.read_csv(csv_path)
        model.tag_names, model.rating_indexes, model.general_indexes, model.character_indexes = self.load_labels(tags_df)

        if model.model:
            del model.model
        model.model = rt.InferenceSession(str(model_path))
        _, height, _, _ = model.model.get_inputs()[0].shape
        model.model_target_size = height
        self.last_loaded_model = model
        print(f"已加载模型: {model.display_name} (仓库: {model.repo_id})")

    def load_labels(self, dataframe):
        name_series = dataframe["name"].map(
            lambda x: x.replace("_", " ") if x not in utils.kaomojis else x
        )
        tag_names = name_series.tolist()
        return (
            tag_names,
            list(np.where(dataframe["category"] == 9)[0]),  # 评分标签
            list(np.where(dataframe["category"] == 0)[0]),  # 通用标签
            list(np.where(dataframe["category"] == 4)[0])   # 角色标签
        )

    def prepare_image(self, image, target_size):
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]
        return np.expand_dims(image_array, axis=0)

    def mcut_threshold(self, probs):
        """MCut自动阈值计算"""
        sorted_probs = probs[probs.argsort()[::-1]]
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        return (sorted_probs[t] + sorted_probs[t + 1]) / 2

    def predict(self, image, model_name, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled):
        model = self.resolve_model(model_name)
        self.load_model(model)
        image = self.prepare_image(image, model.model_target_size)
        input_name = model.model.get_inputs()[0].name
        label_name = model.model.get_outputs()[0].name
        preds = model.model.run([label_name], {input_name: image})[0][0].astype(float)
        labels = list(zip(model.tag_names, preds))
        ratings = dict([labels[i] for i in model.rating_indexes])
        general_tags = [labels[i] for i in model.general_indexes]
        if general_mcut_enabled:
            general_thresh = self.mcut_threshold(np.array([x[1] for x in general_tags]))
        general_tags = dict([x for x in general_tags if x[1] > general_thresh])
        character_tags = [labels[i] for i in model.character_indexes]
        if character_mcut_enabled:
            character_thresh = self.mcut_threshold(np.array([x[1] for x in character_tags]))
            character_thresh = max(0.15, character_thresh)
        character_tags = dict([x for x in character_tags if x[1] > character_thresh])
        sorted_general = sorted(general_tags.items(), key=lambda x: x[1], reverse=True)
        sorted_general_str = ", ".join([x[0] for x in sorted_general]).replace("(", r"\(").replace(")", r"\)")

        return sorted_general_str, ratings, character_tags, general_tags

predictor = Predictor()

def create_gradio_interface():
    TITLE = "WaifuDiffusion Tagger"
    DESCRIPTION = """
    <br>
    - API文档: <code>/docs</code> 或 <code>/redoc</code><br>
    - API接口: <code>/tagger/v1</code>
    """

    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"<h1 style='text-align: center'>{TITLE}</h1>")
        gr.Markdown(DESCRIPTION)
        
        with gr.Row():
            with gr.Column(variant="panel"):
                image = gr.Image(type="pil", image_mode="RGBA", label="输入图像")
                model = gr.Dropdown(
                    choices=utils.all_model_names,
                    label="选择模型",
                    value="wd14-swinv2-v3-git"
                )
                
                with gr.Row():
                    general_thresh = gr.Slider(0, 1, 0.35, 0.05, label="通用标签阈值")
                    general_mcut = gr.Checkbox(False, label="使用MCut自动阈值（通用）")
                
                with gr.Row():
                    char_thresh = gr.Slider(0, 1, 0.85, 0.05, label="角色标签阈值")
                    char_mcut = gr.Checkbox(False, label="使用MCut自动阈值（角色）")
                
                with gr.Row():
                    clear = gr.ClearButton(
                        [image, model, general_thresh, general_mcut, char_thresh, char_mcut],
                        variant="secondary"
                    )
                    submit = gr.Button("提取标签", variant="primary")
            
            with gr.Column(variant="panel"):
                output_str = gr.Textbox(label="格式化标签（逗号分隔）")
                ratings = gr.Label(label="内容评分")
                characters = gr.Label(label="角色识别")
                tags = gr.Label(label="所有标签（带置信度）")

        submit.click(
            fn=predictor.predict,
            inputs=[image, model, general_thresh, general_mcut, char_thresh, char_mcut],
            outputs=[output_str, ratings, characters, tags]
        )

    return demo

# 命令行参数
def parse_args():
    """解析命令行参数，支持host和port配置"""
    parser = argparse.ArgumentParser(description="WaifuDiffusion Tagger 服务")
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="服务绑定的主机地址（例如：127.0.0.1 或 0.0.0.0）"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="服务监听的端口号（例如：7860）"
    )
    parser.add_argument(
        "--score-slider-step", 
        type=float, 
        default=0.05, 
        help="评分滑块的步长"
    )
    parser.add_argument(
        "--score-general-threshold", 
        type=float, 
        default=0.35, 
        help="通用标签的默认阈值"
    )
    parser.add_argument(
        "--score-character-threshold", 
        type=float, 
        default=0.85, 
        help="角色标签的默认阈值"
    )
    return parser.parse_args()

def run_server():
    args = parse_args()

    # 创建FastAPI应用
    app = FastAPI(
        title="WaifuDiffusion Tagger API",
        description="WaifuDiffusion图像标签提取工具的API接口",
        version="1.0.0"
    )

    from api import Api
    Api(app, queue_lock, prefix="/tagger/v1")

    gradio_app = create_gradio_interface()
    app = gr.mount_gradio_app(app, gradio_app, path="")

    print(f"服务启动信息:")
    print(f"- Gradio界面: http://{args.host}:{args.port}")
    print(f"- API文档: http://{args.host}:{args.port}/docs 或 http://{args.host}:{args.port}/redoc")
    print(f"- API接口根路径: http://{args.host}:{args.port}/tagger/v1")
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()