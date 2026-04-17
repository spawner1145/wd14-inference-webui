import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image


MODEL_BASE_DIR = Path("models/taggers")
MODEL_BASE_DIR.mkdir(exist_ok=True, parents=True)

queue_lock = Lock()


@dataclass
class WaifuDiffusionInterrogator:
    display_name: str
    repo_id: str
    revision: str | None = None
    subfolder: str | None = None
    model_type: str = "default"
    use_optimized_model: bool = False
    model_target_size: int | None = None
    model: rt.InferenceSession | None = None
    tag_names: list[str] = field(default_factory=list)
    rating_indexes: list[int] = field(default_factory=list)
    general_indexes: list[int] = field(default_factory=list)
    character_indexes: list[int] = field(default_factory=list)
    artist_indexes: list[int] = field(default_factory=list)
    copyright_indexes: list[int] = field(default_factory=list)
    meta_indexes: list[int] = field(default_factory=list)
    quality_indexes: list[int] = field(default_factory=list)

    def get_model_dir(self, base_dir: Path) -> Path:
        repo_name = self.repo_id.replace("/", "_")
        if self.revision:
            repo_name += f"_{self.revision}"
        if self.subfolder:
            repo_name += f"_{self.subfolder.replace('/', '_')}"
        if self.use_optimized_model:
            repo_name += "_optimized"
        return base_dir / repo_name


class Utils:
    def __init__(self) -> None:
        self.interrogators = {
            "wd14-convnextv2-v2": WaifuDiffusionInterrogator("wd14-convnextv2-v2", repo_id="SmilingWolf/wd-v1-4-convnextv2-tagger-v2", revision="v2.0"),
            "wd14-vit-v2": WaifuDiffusionInterrogator("wd14-vit-v2", repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2", revision="v2.0"),
            "wd14-convnext-v2": WaifuDiffusionInterrogator("wd14-convnext-v2", repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2", revision="v2.0"),
            "wd14-swinv2-v2": WaifuDiffusionInterrogator("wd14-swinv2-v2", repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2", revision="v2.0"),
            "wd14-convnextv2-v2-git": WaifuDiffusionInterrogator("wd14-convnextv2-v2-git", repo_id="SmilingWolf/wd-v1-4-convnextv2-tagger-v2"),
            "wd14-vit-v2-git": WaifuDiffusionInterrogator("wd14-vit-v2-git", repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2"),
            "wd14-convnext-v2-git": WaifuDiffusionInterrogator("wd14-convnext-v2-git", repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2"),
            "wd14-swinv2-v2-git": WaifuDiffusionInterrogator("wd14-swinv2-v2-git", repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2"),
            "wd14-vit": WaifuDiffusionInterrogator("wd14-vit", repo_id="SmilingWolf/wd-v1-4-vit-tagger"),
            "wd14-convnext": WaifuDiffusionInterrogator("wd14-convnext", repo_id="SmilingWolf/wd-v1-4-convnext-tagger"),
            "wd14-vit-v3-git": WaifuDiffusionInterrogator("wd14-vit-v3-git", repo_id="SmilingWolf/wd-vit-tagger-v3"),
            "wd14-convnext-v3-git": WaifuDiffusionInterrogator("wd14-convnext-v3-git", repo_id="SmilingWolf/wd-convnext-tagger-v3"),
            "wd14-swinv2-v3-git": WaifuDiffusionInterrogator("wd14-swinv2-v3-git", repo_id="SmilingWolf/wd-swinv2-tagger-v3"),
            "wd14-large-v3-git": WaifuDiffusionInterrogator("wd14-large-v3-git", repo_id="SmilingWolf/wd-vit-large-tagger-v3"),
            "wd14-eva02-large-v3-git": WaifuDiffusionInterrogator("wd14-eva02-large-v3-git", repo_id="SmilingWolf/wd-eva02-large-tagger-v3"),
            "idolsankaku-swinv2-tagger-v1": WaifuDiffusionInterrogator("idolsankaku-swinv2-tagger-v1", repo_id="deepghs/idolsankaku-swinv2-tagger-v1"),
            "idolsankaku-eva02-large-tagger-v1": WaifuDiffusionInterrogator("idolsankaku-eva02-large-tagger-v1", repo_id="deepghs/idolsankaku-eva02-large-tagger-v1"),
            "wd-v1-4-moat-tagger-v2": WaifuDiffusionInterrogator("wd-v1-4-moat-tagger-v2", repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2"),
            "cltagger-v1.02": WaifuDiffusionInterrogator("cltagger-v1.02", repo_id="cella110n/cl_tagger", subfolder="cl_tagger_1_02", model_type="cl_tagger"),
            "cltagger-v1.02-optimized": WaifuDiffusionInterrogator("cltagger-v1.02-optimized", repo_id="cella110n/cl_tagger", subfolder="cl_tagger_1_02", model_type="cl_tagger", use_optimized_model=True),
        }
        self.builtin_model_names = list(self.interrogators.keys())
        self.kaomojis = [
            "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=",
            ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
        ]
        self.all_model_names = list(self.interrogators.keys())

    def _register_local_model(
        self,
        model_name: str,
        model_dir: Path,
        model_type: str,
        use_optimized_model: bool = False,
    ) -> None:
        self.interrogators[model_name] = WaifuDiffusionInterrogator(
            display_name=model_name,
            repo_id=f"local/{model_name}",
            model_type=model_type,
            use_optimized_model=use_optimized_model,
        )
        self.interrogators[model_name].get_model_dir = lambda _base_dir, target=model_dir: target

    def refresh_interrogators(self) -> list[str]:
        for model_dir in MODEL_BASE_DIR.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            if model_name in self.interrogators:
                continue

            default_model = model_dir / "model.onnx"
            optimized_model = model_dir / "model_optimized.onnx"
            tag_csv = model_dir / "selected_tags.csv"
            tag_json = model_dir / "tag_mapping.json"

            if default_model.exists() and tag_csv.exists():
                self._register_local_model(model_name, model_dir, "default")
            elif default_model.exists() and tag_json.exists():
                self._register_local_model(model_name, model_dir, "cl_tagger")
            elif optimized_model.exists() and tag_json.exists():
                self._register_local_model(model_name, model_dir, "cl_tagger", use_optimized_model=True)

        self.all_model_names = list(self.interrogators.keys())
        return self.all_model_names


utils = Utils()
utils.refresh_interrogators()


class Predictor:
    def __init__(self) -> None:
        self.last_loaded_model: WaifuDiffusionInterrogator | None = None

    def resolve_model(self, model_name: str) -> WaifuDiffusionInterrogator:
        if model_name not in utils.interrogators:
            available = ", ".join(utils.interrogators.keys())
            raise ValueError(f"Model not found: {model_name}. Available models: {available}")
        return utils.interrogators[model_name]

    def download_model(self, model: WaifuDiffusionInterrogator) -> tuple[Path, Path]:
        model_dir = model.get_model_dir(MODEL_BASE_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_filename = "model.onnx"
        label_csv = "selected_tags.csv"
        label_json = "tag_mapping.json"

        if model.model_type == "cl_tagger":
            model_filename = "model_optimized.onnx" if model.use_optimized_model else "model.onnx"
            allowed_patterns = [f"{model.subfolder}/{name}" for name in (model_filename, label_json)]
        else:
            allowed_patterns = [f"{model.subfolder}/{name}" if model.subfolder else name for name in (model_filename, label_csv)]

        model_path = model_dir / model_filename
        label_path_csv = model_dir / label_csv
        label_path_json = model_dir / label_json

        if not model_path.exists() or not (label_path_csv.exists() or label_path_json.exists()):
            temp_dir = model_dir / "temp_download"
            try:
                huggingface_hub.snapshot_download(
                    repo_id=model.repo_id,
                    revision=model.revision,
                    local_dir=temp_dir,
                    allow_patterns=allowed_patterns,
                    local_dir_use_symlinks=False,
                )
                source_dir = temp_dir / model.subfolder if model.subfolder else temp_dir
                for source, target in (
                    (source_dir / model_filename, model_path),
                    (source_dir / label_csv, label_path_csv),
                    (source_dir / label_json, label_path_json),
                ):
                    if source.exists():
                        shutil.move(str(source), str(target))
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

        if label_path_csv.exists():
            return label_path_csv, model_path
        if label_path_json.exists():
            return label_path_json, model_path
        raise FileNotFoundError(f"Tag metadata not found in {model_dir}")

    def load_model(self, model: WaifuDiffusionInterrogator) -> None:
        if model == self.last_loaded_model:
            return

        with queue_lock:
            if model == self.last_loaded_model:
                return

            label_path, model_path = self.download_model(model)
            self._reset_labels(model)
            if model.model_type == "cl_tagger":
                self.load_labels_cl(model, label_path)
            else:
                self.load_labels_default(model, label_path)

            if model.model is not None:
                del model.model

            model.model = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            if model.model_type == "cl_tagger":
                model.model_target_size = 448
            else:
                model.model_target_size = int(model.model.get_inputs()[0].shape[1])

            self.last_loaded_model = model

    def _reset_labels(self, model: WaifuDiffusionInterrogator) -> None:
        model.tag_names = []
        model.rating_indexes = []
        model.general_indexes = []
        model.character_indexes = []
        model.artist_indexes = []
        model.copyright_indexes = []
        model.meta_indexes = []
        model.quality_indexes = []

    def load_labels_default(self, model: WaifuDiffusionInterrogator, csv_path: Path) -> None:
        dataframe = pd.read_csv(csv_path)
        model.tag_names = dataframe["name"].map(
            lambda value: value.replace("_", " ") if value not in utils.kaomojis else value
        ).tolist()
        model.rating_indexes = list(np.where(dataframe["category"] == 9)[0])
        model.general_indexes = list(np.where(dataframe["category"] == 0)[0])
        model.character_indexes = list(np.where(dataframe["category"] == 4)[0])

    def load_labels_cl(self, model: WaifuDiffusionInterrogator, json_path: Path) -> None:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if "idx_to_tag" in data:
            index_map = {int(key): value for key, value in data["idx_to_tag"].items()}
            category_map = data["tag_to_category"]
        else:
            entries = {int(key): value for key, value in data.items()}
            index_map = {index: value["tag"] for index, value in entries.items()}
            category_map = {value["tag"]: value["category"] for value in entries.values()}

        model.tag_names = [""] * (max(index_map.keys()) + 1)
        for index, tag in index_map.items():
            model.tag_names[index] = tag.replace("_", " ")
            category = category_map.get(tag, "Unknown")
            if category == "Rating":
                model.rating_indexes.append(index)
            elif category == "General":
                model.general_indexes.append(index)
            elif category == "Artist":
                model.artist_indexes.append(index)
            elif category == "Character":
                model.character_indexes.append(index)
            elif category == "Copyright":
                model.copyright_indexes.append(index)
            elif category == "Meta":
                model.meta_indexes.append(index)
            elif category == "Quality":
                model.quality_indexes.append(index)

    def prepare_image_default(self, image: Image.Image, target_size: int) -> np.ndarray:
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))
        if max_dim != target_size:
            padded = padded.resize((target_size, target_size), Image.BICUBIC)
        image_array = np.asarray(padded, dtype=np.float32)[:, :, ::-1]
        return np.expand_dims(image_array, axis=0)

    def prepare_image_cl(self, image: Image.Image, target_size: int) -> np.ndarray:
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGB")
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        width, height = image.size
        if width != height:
            edge = max(width, height)
            padded = Image.new("RGB", (edge, edge), (255, 255, 255))
            padded.paste(image, ((edge - width) // 2, (edge - height) // 2))
            image = padded
        image_array = np.array(image.resize((target_size, target_size), Image.BICUBIC), dtype=np.float32) / 255.0
        image_array = image_array.transpose(2, 0, 1)
        mean = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1)
        image_array = (image_array - mean) / std
        return np.expand_dims(image_array, axis=0)

    def mcut_threshold(self, probs: np.ndarray) -> float:
        sorted_probs = probs[probs.argsort()[::-1]]
        diffs = sorted_probs[:-1] - sorted_probs[1:]
        threshold_index = int(diffs.argmax())
        return float((sorted_probs[threshold_index] + sorted_probs[threshold_index + 1]) / 2)

    def get_tags_cl(
        self,
        probs: np.ndarray,
        model: WaifuDiffusionInterrogator,
        general_threshold: float,
        character_threshold: float,
    ) -> dict[str, list[tuple[str, float]]]:
        predictions: dict[str, list[tuple[str, float]]] = {}
        if model.rating_indexes:
            predictions["rating"] = [(model.tag_names[index], float(probs[index])) for index in model.rating_indexes]

        if model.quality_indexes:
            quality_probs = probs[model.quality_indexes]
            best_index = int(np.argmax(quality_probs))
            predictions["quality"] = [(model.tag_names[model.quality_indexes[best_index]], float(quality_probs[best_index]))]

        category_map = {
            "general": (model.general_indexes, general_threshold),
            "character": (model.character_indexes, character_threshold),
            "copyright": (model.copyright_indexes, character_threshold),
            "artist": (model.artist_indexes, character_threshold),
            "meta": (model.meta_indexes, general_threshold),
        }
        for category, (indexes, threshold) in category_map.items():
            if indexes:
                predictions[category] = sorted(
                    [
                        (model.tag_names[index], float(probs[index]))
                        for index in indexes
                        if probs[index] > threshold
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )
        return predictions

    def predict(
        self,
        image: Image.Image,
        model_name: str,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
    ) -> tuple[str, dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        model = self.resolve_model(model_name)
        self.load_model(model)
        assert model.model is not None
        assert model.model_target_size is not None

        if model.model_type == "cl_tagger":
            image_tensor = self.prepare_image_cl(image, model.model_target_size)
        else:
            image_tensor = self.prepare_image_default(image, model.model_target_size)

        input_name = model.model.get_inputs()[0].name
        output_name = model.model.get_outputs()[0].name
        preds = model.model.run([output_name], {input_name: image_tensor.astype(np.float32)})[0][0]

        if model.model_type == "cl_tagger":
            probs = 1 / (1 + np.exp(-np.clip(preds, -30, 30)))
            predictions = self.get_tags_cl(probs, model, general_thresh, character_thresh)
            ratings = dict(predictions.get("rating", []))
            quality = dict(predictions.get("quality", []))
            characters = dict(predictions.get("character", []))
            artist = dict(predictions.get("artist", []))
            copyright_tags = dict(predictions.get("copyright", []))
            meta = dict(predictions.get("meta", []))
            general_tags = dict(predictions.get("general", []))
            combined = {**general_tags, **characters, **copyright_tags, **artist, **meta}
            sorted_tags = sorted(combined.items(), key=lambda item: item[1], reverse=True)
            caption = ",".join(tag for tag, _ in sorted_tags).replace("(", r"\(").replace(")", r"\)")
            return caption, ratings, quality, characters, artist, copyright_tags, meta, general_tags

        labels = list(zip(model.tag_names, preds.astype(float)))
        ratings = dict(labels[index] for index in model.rating_indexes)

        general_tags_list = [labels[index] for index in model.general_indexes]
        if general_mcut_enabled and len(general_tags_list) > 1:
            general_thresh = self.mcut_threshold(np.array([value for _, value in general_tags_list]))
        general_tags = dict(item for item in general_tags_list if item[1] > general_thresh)

        character_tags_list = [labels[index] for index in model.character_indexes]
        if character_mcut_enabled and len(character_tags_list) > 1:
            character_thresh = self.mcut_threshold(np.array([value for _, value in character_tags_list]))
        character_tags = dict(item for item in character_tags_list if item[1] > character_thresh)

        combined = {**general_tags, **character_tags}
        sorted_tags = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        caption = ",".join(tag for tag, _ in sorted_tags).replace("(", r"\(").replace(")", r"\)")
        return caption, ratings, {}, character_tags, {}, {}, {}, combined


predictor = Predictor()
