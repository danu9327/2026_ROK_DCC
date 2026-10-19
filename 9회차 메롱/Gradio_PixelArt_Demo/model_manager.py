"""
model_manager.py
Image-to-Image 모델 로딩/언로딩 및 추론 관리
VRAM 효율을 위해 한 번에 하나의 파이프라인만 유지
"""

import gc
import math
import threading
from typing import Optional

import torch
from diffusers import (
    AutoencoderKL,
    AutoPipelineForImage2Image,
    ControlNetModel,
    FluxKontextPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from PIL import Image

from pixel_utils import canny_image, fit_image

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"   # 기본 SDXL VAE는 fp16에서 깨짐(NaN)
PIXEL_LORA_REPO = "nerijs/pixel-art-xl"
PIXEL_LORA_FILE = "pixel-art-xl.safetensors"
CANNY_CONTROLNET = "diffusers/controlnet-canny-sdxl-1.0"
SDXL_TURBO = "stabilityai/sdxl-turbo"
FLUX_KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"   # gated: HF에서 라이선스 동의 필요
KONTEXT_LORA_REPO = "Kontext-Style/Pixel_lora"
KONTEXT_LORA_FILE = "Pixel_lora_weights.safetensors"
KONTEXT_TRIGGER = "Turn this image into the Pixel style."

# ── 모델 메타데이터 ──────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "하 - SDXL-Turbo (프롬프트만)": {
        "kind": "turbo",
        "model_id": SDXL_TURBO,
        "size": 512,
        "steps": 4,
        "guidance": 0.0,
        "description": (
            "🔵 **SDXL-Turbo**\n"
            "- 4스텝 초고속 생성 (1~2초)\n"
            "- 픽셀 스타일은 **프롬프트로만** 유도\n"
            "- Guidance 0 고정 → 부정 프롬프트 무시됨\n"
            "- VRAM: ~8GB"
        ),
    },
    "중 - SDXL + Pixel-Art LoRA": {
        "kind": "sdxl",
        "model_id": SDXL_BASE,
        "size": 1024,
        "steps": 30,
        "guidance": 7.0,
        "description": (
            "🟡 **SDXL + Pixel-Art-XL LoRA**\n"
            "- 픽셀아트 데이터로 학습된 LoRA 어댑터 장착\n"
            "- 프롬프트만 쓸 때보다 훨씬 '진짜 도트' 느낌\n"
            "- 1024px, 10~15초\n"
            "- VRAM: ~12GB"
        ),
    },
    "상 - SDXL + LoRA + ControlNet": {
        "kind": "sdxl_cn",
        "model_id": SDXL_BASE,
        "size": 1024,
        "steps": 30,
        "guidance": 7.0,
        "description": (
            "🔴 **SDXL + LoRA + ControlNet(Canny)**\n"
            "- 원본의 **윤곽선(엣지)**을 뼈대로 고정\n"
            "- 변형 강도를 높여도 원래 형태/포즈 유지\n"
            "- 1024px, 15~20초\n"
            "- VRAM: ~16GB"
        ),
    },
    "최상 - FLUX Kontext + Pixel LoRA": {
        "kind": "kontext",
        "model_id": FLUX_KONTEXT,
        "size": 1024,
        "steps": 24,
        "guidance": 2.5,
        "description": (
            "🟣 **FLUX.1 Kontext + Pixel LoRA**\n"
            "- 노이즈를 섞지 않고 **문장 지시로 편집** (\"픽셀 스타일로 바꿔줘\")\n"
            "- 얼굴·구도·포즈 보존력 최고 → **변형 강도 설정 없음**\n"
            "- 12B 대형 모델, 1024px, 30~60초\n"
            "- VRAM: ~24GB (CPU 오프로드 사용)"
        ),
    },
}


# ── PixelArtManager 클래스 ──────────────────────────────────────────────────

class PixelArtManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.lock = threading.Lock()   # 탭 간 동시 요청 시 모델 교체 충돌 방지
        self._pipe = None
        self._name: Optional[str] = None

    def _clear(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._name = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def vram_info(self) -> str:
        if not torch.cuda.is_available():
            return "GPU 없음 (CPU 모드)"
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"VRAM: {alloc:.1f}GB / {total:.0f}GB 사용 중"

    def load(self, name: str, status_fn=None):
        if self._name == name:
            return
        cfg = MODEL_CONFIGS[name]
        if status_fn:
            status_fn(f"⏳ 모델 로딩 중: {name} (최초 1회 다운로드 시 수 분 소요)")

        self._clear()
        variant = "fp16" if self.dtype == torch.float16 else None

        if cfg["kind"] == "kontext":
            pipe = FluxKontextPipeline.from_pretrained(cfg["model_id"], torch_dtype=torch.bfloat16)
            pipe.load_lora_weights(KONTEXT_LORA_REPO, weight_name=KONTEXT_LORA_FILE, adapter_name="pixel")
        elif cfg["kind"] == "turbo":
            pipe = AutoPipelineForImage2Image.from_pretrained(
                cfg["model_id"], torch_dtype=self.dtype, variant=variant
            )
        else:
            vae = AutoencoderKL.from_pretrained(SDXL_FP16_VAE, torch_dtype=self.dtype)
            if cfg["kind"] == "sdxl_cn":
                controlnet = ControlNetModel.from_pretrained(
                    CANNY_CONTROLNET, torch_dtype=self.dtype, variant=variant
                )
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    cfg["model_id"], controlnet=controlnet, vae=vae,
                    torch_dtype=self.dtype, variant=variant, use_safetensors=True,
                )
            else:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    cfg["model_id"], vae=vae,
                    torch_dtype=self.dtype, variant=variant, use_safetensors=True,
                )
            pipe.load_lora_weights(PIXEL_LORA_REPO, weight_name=PIXEL_LORA_FILE, adapter_name="pixel")

        if cfg["kind"] == "kontext" and self.device == "cuda":
            # 전체 ~34GB(transformer 24GB + T5 10GB)라 32GB에 다 못 올림 → 쓰는 부품만 GPU로
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._name = name

        if status_fn:
            status_fn(f"✅ 모델 로드 완료 | {self.vram_info()}")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance: float,
        seed: int,
        lora_scale: float = 1.0,
        controlnet_scale: float = 0.6,
    ) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        cfg = MODEL_CONFIGS[self._name]
        init = fit_image(image, cfg["size"])

        kwargs = dict(
            prompt=prompt,
            image=init,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        if cfg["kind"] == "kontext":
            kwargs.update(width=init.width, height=init.height)
        else:
            # img2img는 실제로 steps × strength 만큼만 디노이징 → 최소 1스텝 보장
            kwargs["num_inference_steps"] = max(int(steps), math.ceil(1 / strength))
            kwargs["strength"] = strength
        if cfg["kind"] != "turbo":
            self._pipe.set_adapters(["pixel"], adapter_weights=[lora_scale])
        if cfg["kind"] in ("sdxl", "sdxl_cn"):
            kwargs["negative_prompt"] = negative_prompt
        if cfg["kind"] == "sdxl_cn":
            kwargs["control_image"] = canny_image(init)
            kwargs["controlnet_conditioning_scale"] = controlnet_scale

        # inference_mode 금지: CPU 오프로드(Kontext) 시 가중치가 inference tensor로 바뀌어
        # 다음 호출의 set_adapters(requires_grad 변경)가 실패함. 파이프라인 자체가 no_grad로 동작
        return self._pipe(**kwargs).images[0]


# 전역 싱글턴
manager = PixelArtManager()
