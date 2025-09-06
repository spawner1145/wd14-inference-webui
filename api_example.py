import requests
import base64
import json
from PIL import Image
from io import BytesIO
import os

class TaggerAPIClient:
    def __init__(self, base_url: str):
        """初始化API客户端"""
        self.base_url = base_url.rstrip('/')
        self.interrogate_endpoint = f"{self.base_url}/interrogate"
        self.interrogators_endpoint = f"{self.base_url}/interrogators"
        
    def get_available_models(self):
        """获取所有可用的模型列表及信息"""
        try:
            response = requests.get(self.interrogators_endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"获取模型列表失败: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应内容: {e.response.text}")
            return None
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片文件编码为base64字符串"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGBA格式以匹配API预期
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # 保存到内存缓冲区
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                
                # 编码为base64
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return img_str
        except Exception as e:
            print(f"图片编码失败: {str(e)}")
            return None
    
    def interrogate_image(self, 
                         image_path: str, 
                         model: str, 
                         threshold: float = 0.35,
                         character_threshold: float = 0.85,
                         general_mcut_enabled: bool = False,
                         character_mcut_enabled: bool = False):
        """调用interrogate接口分析图片"""
        # 编码图片
        image_base64 = self.encode_image_to_base64(image_path)
        if not image_base64:
            return None
            
        # 构建请求数据
        payload = {
            "image": image_base64,
            "model": model,
            "threshold": threshold,
            "character_threshold": character_threshold,
            "general_mcut_enabled": general_mcut_enabled,
            "character_mcut_enabled": character_mcut_enabled
        }
        
        try:
            response = requests.post(
                self.interrogate_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"图片分析请求失败: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            return None

def main():
    # 硬编码配置 - 在这里修改参数
    API_BASE_URL = "http://127.0.0.1:7860/tagger/v1"  # API服务地址
    IMAGE_PATH = "ComfyUI_00082_.png"          # 要分析的图片路径
    MODEL_NAME = "cltagger-v1.02"           # 要使用的模型名称
    THRESHOLD = 0.35                       # 通用标签阈值
    CHARACTER_THRESHOLD = 0.85             # 角色标签阈值
    ENABLE_GENERAL_MCUT = False            # 是否启用通用标签MCUT
    ENABLE_CHARACTER_MCUT = False          # 是否启用角色标签MCUT
    LIST_MODELS_FIRST = True               # 是否先列出所有模型
    
    # 创建客户端实例
    client = TaggerAPIClient(API_BASE_URL)
    
    # 先列出所有可用模型（如果启用）
    if LIST_MODELS_FIRST:
        print("获取可用模型列表...")
        models_info = client.get_available_models()
        if models_info:
            print("\n可用模型:")
            for model_name in models_info['models']:
                info = models_info['model_info'][model_name]
                print(f"- {model_name}:")
                print(f"  仓库ID: {info['repo_id']}")
                print(f"  版本: {info['revision']}")
                print(f"  子文件夹: {info['subfolder'] or '无'}")
                print(f"  模型类型: {info['model_type']}")
        print("\n" + "="*50 + "\n")
    
    # 检查图片文件是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"错误: 图片文件不存在 - {IMAGE_PATH}")
        return
    
    # 调用API分析图片
    print(f"使用模型 {MODEL_NAME} 分析图片 {IMAGE_PATH}...")
    result = client.interrogate_image(
        image_path=IMAGE_PATH,
        model=MODEL_NAME,
        threshold=THRESHOLD,
        character_threshold=CHARACTER_THRESHOLD,
        general_mcut_enabled=ENABLE_GENERAL_MCUT,
        character_mcut_enabled=ENABLE_CHARACTER_MCUT
    )
    
    # 处理结果
    if result:
        print("\n分析结果:")
        print(f"使用的模型: {result['model_used']}")
        print("\n识别到的标签 (按置信度排序):")
        
        # 打印前20个标签
        tags = list(result['caption'].items())
        for i, (tag, score) in enumerate(tags[:20]):
            print(f"  {tag}: {score:.4f}")
        
        if len(tags) > 20:
            print(f"  ... 还有 {len(tags) - 20} 个标签未显示")

if __name__ == "__main__":
    main()
    
