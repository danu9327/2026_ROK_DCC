"""
download_models.py
수업 전에 모델을 미리 받아두는 스크립트 (HF 캐시: ~/.cache/huggingface/hub)

사용법:
    python download_models.py                  # 전체 (하/중/상/최상)
    python download_models.py 하 중            # 원하는 등급만
"""

import argparse

from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download

from model_manager import (
    CANNY_CONTROLNET, FLUX_KONTEXT, KONTEXT_LORA_FILE, KONTEXT_LORA_REPO,
    PIXEL_LORA_FILE, PIXEL_LORA_REPO, SDXL_BASE, SDXL_FP16_VAE, SDXL_TURBO,
)


def turbo():
    DiffusionPipeline.download(SDXL_TURBO, variant="fp16")


def sdxl():
    DiffusionPipeline.download(SDXL_BASE, variant="fp16", use_safetensors=True)
    snapshot_download(SDXL_FP16_VAE, allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
    hf_hub_download(PIXEL_LORA_REPO, PIXEL_LORA_FILE)


def controlnet():
    sdxl()
    snapshot_download(CANNY_CONTROLNET, allow_patterns=["config.json", "diffusion_pytorch_model.fp16.safetensors"])


def kontext():
    DiffusionPipeline.download(FLUX_KONTEXT)
    hf_hub_download(KONTEXT_LORA_REPO, KONTEXT_LORA_FILE)


TIERS = {
    "하": ("SDXL-Turbo (~7GB)", turbo),
    "중": ("SDXL base + fp16 VAE + Pixel-Art-XL LoRA (~7GB)", sdxl),
    "상": ("중 + ControlNet Canny (~+2.5GB)", controlnet),
    "최상": ("FLUX.1 Kontext + Pixel LoRA (~34GB, 라이선스 동의 필요)", kontext),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tiers", nargs="*", choices=list(TIERS), default=list(TIERS))
    args = parser.parse_args()

    for t in args.tiers:
        desc, fn = TIERS[t]
        print(f"\n⏳ [{t}] {desc} 다운로드 중...")
        try:
            fn()
            print(f"✅ [{t}] 완료")
        except Exception as e:
            print(f"❌ [{t}] 실패: {e}")
            if t == "최상":
                print("   → https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev 에서 라이선스 동의 후 "
                      "`hf auth login` 으로 같은 계정 토큰을 등록했는지 확인하세요.")
