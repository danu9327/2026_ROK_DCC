"""
model_manager.py
모델 로딩/언로딩 및 추론 관리
VRAM 효율을 위해 한 번에 하나의 텍스트 모델, 하나의 이미지 모델만 유지
"""

import gc
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

# ── 모델 메타데이터 ──────────────────────────────────────────────────────────

TEXT_MODEL_CONFIGS = {
    "하 - Phi-3.5-mini (3.8B)": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "description": (
            "🔵 **소형 모델 (3.8B)**\n"
            "- 빠른 응답 속도\n"
            "- 한국어 → 영어 변환 + 기본 프롬프트 확장\n"
            "- VRAM: ~6GB"
        ),
        "load_in_4bit": False,
        "torch_dtype": torch.bfloat16,
    },
    "중 - Qwen2.5-7B (7B)": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": (
            "🟡 **중형 모델 (7B)**\n"
            "- 균형잡힌 속도/품질\n"
            "- 복잡한 개념 이해, 풍부한 묘사 생성\n"
            "- VRAM: ~14GB"
        ),
        "load_in_4bit": False,
        "torch_dtype": torch.bfloat16,
    },
    "상 - Qwen2.5-14B (14B)": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "description": (
            "🔴 **대형 모델 (14B, 4-bit 양자화)**\n"
            "- 최고 수준의 텍스트 이해\n"
            "- 예술적 방향, 세부 묘사 탁월\n"
            "- VRAM: ~10GB (4-bit)"
        ),
        "load_in_4bit": True,
        "torch_dtype": torch.bfloat16,
    },
}

IMAGE_MODEL_CONFIGS = {
    "하 - SD v1.5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "description": (
            "🔵 **Stable Diffusion v1.5**\n"
            "- 512×512 기본 해상도\n"
            "- LoRA 파인튜닝 가능! (학습 탭에서 체험)\n"
            "- VRAM: ~4GB"
        ),
        "pipeline_cls": "SD15",
        "default_size": (512, 512),
        "supports_negative": True,
    },
    "중 - SDXL": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "description": (
            "🟡 **Stable Diffusion XL**\n"
            "- 1024×1024 고해상도\n"
            "- 더 세밀한 디테일, 높은 품질\n"
            "- VRAM: ~10GB"
        ),
        "pipeline_cls": "SDXL",
        "default_size": (1024, 1024),
        "supports_negative": True,
    },
    "상 - Playground v2.5": {
        "model_id": "playgroundai/playground-v2.5-1024px-aesthetic",
        "description": (
            "🔴 **Playground v2.5 (고품질 모델)**\n"
            "- 1024×1024 최고 해상도\n"
            "- 미적 품질 특화, 무인증 모델\n"
            "- VRAM: ~10GB"
        ),
        "pipeline_cls": "SDXL",
        "default_size": (1024, 1024),
        "supports_negative": True,
    },
}

PROMPT_SYSTEM = """You are an expert AI image prompt engineer.
Your task is to take the user's input (which may be in Korean or simple English)
and convert it into a detailed, high-quality English prompt suitable for AI image generation.

Rules:
1. Output ONLY the enhanced English prompt, nothing else
2. Add artistic details: lighting, style, composition, quality descriptors
3. Keep it under 200 words
4. Include: subject description, style, lighting, camera angle, quality tags"""


# ── ModelManager 클래스 ─────────────────────────────────────────────────────

class ModelManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._text_model = None
        self._text_tokenizer = None
        self._text_model_id: Optional[str] = None

        self._image_pipe = None
        self._image_model_id: Optional[str] = None
        self._image_pipeline_cls: Optional[str] = None

    # ── 메모리 정리 ──────────────────────────────────────────────────────────

    def _clear_text(self):
        if self._text_model is not None:
            del self._text_model, self._text_tokenizer
            self._text_model = self._text_tokenizer = None
            self._text_model_id = None
            self._flush()

    def _clear_image(self):
        if self._image_pipe is not None:
            del self._image_pipe
            self._image_pipe = None
            self._image_model_id = None
            self._image_pipeline_cls = None
            self._flush()

    def _flush(self):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def vram_info(self) -> str:
        if not torch.cuda.is_available():
            return "GPU 없음 (CPU 모드)"
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"VRAM: {alloc:.1f}GB / {total:.0f}GB 사용 중"

    # ── 텍스트 모델 ──────────────────────────────────────────────────────────

    def load_text_model(self, config_name: str, status_fn=None):
        cfg = TEXT_MODEL_CONFIGS[config_name]
        model_id = cfg["model_id"]

        if self._text_model_id == model_id:
            return

        if status_fn:
            status_fn(f"⏳ 텍스트 모델 로딩 중: {model_id}")

        self._clear_text()

        if cfg["load_in_4bit"]:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=cfg["torch_dtype"], device_map="auto"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._text_model = model
        self._text_tokenizer = tokenizer
        self._text_model_id = model_id

        if status_fn:
            status_fn(f"✅ 텍스트 모델 로드 완료 | {self.vram_info()}")

    def enhance_prompt(self, user_input: str) -> str:
        if self._text_model is None:
            raise RuntimeError("텍스트 모델이 로드되지 않았습니다.")

        messages = [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_input},
        ]
        text = self._text_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._text_tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self._text_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._text_tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._text_tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ── 이미지 모델 ──────────────────────────────────────────────────────────

    def load_image_model(self, config_name: str, status_fn=None):
        cfg = IMAGE_MODEL_CONFIGS[config_name]
        model_id = cfg["model_id"]

        if self._image_model_id == model_id:
            return

        if status_fn:
            status_fn(f"⏳ 이미지 모델 로딩 중: {model_id}")

        self._clear_image()
        cls = cfg["pipeline_cls"]

        if cls == "SD15":
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, safety_checker=None
            )
        elif cls == "SDXL":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, use_safetensors=True
            )
        else:
            raise ValueError(f"알 수 없는 파이프라인: {cls}")

        pipe = pipe.to(self.device)
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self._image_pipe = pipe
        self._image_model_id = model_id
        self._image_pipeline_cls = cls

        if status_fn:
            status_fn(f"✅ 이미지 모델 로드 완료 | {self.vram_info()}")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = -1,
        lora_path: Optional[str] = None,
    ) -> Image.Image:
        if self._image_pipe is None:
            raise RuntimeError("이미지 모델이 로드되지 않았습니다.")

        # LoRA 적용: PEFT 방식으로 UNet에 직접 장착 후 생성, 이후 복원
        original_unet = None
        if lora_path and self._image_pipeline_cls == "SD15":
            from peft import PeftModel
            original_unet = self._image_pipe.unet
            lora_unet = PeftModel.from_pretrained(
                original_unet, lora_path, is_trainable=False
            )
            self._image_pipe.unet = lora_unet

        try:
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            with torch.no_grad():
                result = self._image_pipe(**kwargs)
        finally:
            # LoRA UNet 복원
            if original_unet is not None:
                self._image_pipe.unet = original_unet
                del lora_unet
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        return result.images[0]


# 전역 싱글턴
manager = ModelManager()
